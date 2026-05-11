### 1. High-Level Architecture

```
                          ┌──────────────────────────────┐
                          │        GROUP JSON INPUT        │
                          │  (courses, grades, interests,  │
                          │   applications, RDIA choice,   │
                          │   optional weighting_mode)     │
                          └──────────────┬───────────────┘
                                         │
                                         ▼
                          ┌──────────────────────────────┐
                          │       EmbeddingEngine         │
                          │  (shared, initialized once)   │
                          │  • SBERT model                │
                          │  • Project matrix (N×D)       │
                          │  • BM25 index                 │
                          │  • Pre-computed domain vecs   │
                          │  • Pre-computed PLO vecs      │
                          │  • build_group_profile()      │
                          └──────────────┬───────────────┘
                                         │  group_vec + group_meta
                        ┌────────────────┼──────────────────┼──────────────────┐
                        │                │                  │                  │
                        ▼                ▼                  ▼                  ▼
              ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         ┌──────────────┐ 
              │   Project    │  │   Interest   │  │ Application  │         │     RDIA     │
              │  Recommender │  │  Recommender │  │  Recommender │         │  Recommender │
              └──────────────┘  └──────────────┘  └──────────────┘         └──────────────┘
                     |────────────────┼──────────────────┼─────────────────────────────|
            
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │  UNIFIED OUTPUT │
                                                └─────────────────┘
```
---

## 2. Embedding Strategy

| Entity | What the Embedding Captures |
|---|---|
| Course | The knowledge domain of a course, derived from its title, description, and CLO statements (each CLO linked to its PLO description) |
| Project | The full intellectual content of a past project — its problem, goals, methods, and outcomes |
| Interest domain | The conceptual meaning of a named interest category (e.g., "Computer Vision") |
| Application domain | The focus area of a target field (e.g., "Healthcare / Medical") |
| RDIA priority | The thematic alignment of a Saudi research initiative |
| Student profile | A weighted blend of the student's competency and interest vectors |
| Group profile | A mean-pooled aggregation of all individual student vectors |


### 1. The SBERT Model

The system uses `all-MiniLM-L6-v2`.


### 2. Late Fusion Encoding

The core embedding strategy is Late Fusion — encoding each meaningful text component separately, then averaging the resulting vectors into one composite representation.

**Why Late Fusion instead of text concatenation?**

SBERT has a hard 512-token input limit. A long project description concatenated with its keywords, objectives, abstract, and domain labels would be truncated — effectively discarding the later components. Late Fusion avoids this by giving each component its own 512-token budget.

**Segments encoded per project:**

1. Title
2. Abstract
3. Keywords (joined)
4. Problem statement + Aim (joined)
5. Objectives (joined)
6. Results
7. Future work
8. Domain labels (application + interest + RDIA joined)
9. ACM taxonomy descriptions

**Segments encoded per course:**

1. Title + Description (one segment)
2. Each CLO statement individually, enriched with its linked PLO description

After encoding all segments, their vectors are averaged and re-normalized. This produces a single vector that captures all components of the entity with equal weight.

### 3. Precomputation Strategy

All static content (courses, projects, domain options) is embedded once during an offline Phase II step and stored as `.npy` files.

### 4. Grade Weighting

Each student's course contributions are weighted by their grade according to a fixed mapping:

| Grade | Weight |
|---|---|
| A+ | 1.00 |
| A | 0.95 |
| B+ | 0.85 |
| B | 0.75 |
| C+ | 0.65 |
| C | 0.55 |
| D+ | 0.45 |
| D | 0.30 |

Unknown or missing grades default to a weight of **0.50**.

### 5. Vector Normalization

All vectors — after averaging, weighted averaging, or mean-pooling — are L2-normalized.

---

## 3. Recommendation Logic

### 3.1 Building the Group Profile

**Weighting Mode**

The group JSON may include an optional `weighting_mode` field that controls how much the academic background (competency) versus declared preferences (interests) influence the group vector:

| Mode | Competency Weight | Interest Weight |
|---|---|---|
| `balanced` (default) | 0.50 | 0.50 |
| `courses_heavy` | 0.75 | 0.25 |
| `interests_heavy` | 0.25 | 0.75 |

**Stage 1 — Per-Student Competency Vector**

Each course the student has taken contributes a pre-computed embedding vector, weighted by grade. These are combined into a single competency vector using weighted averaging, then normalized.

```
competency_vec = normalize( Σ weight_i × course_vec_i )
```

If a course embedding is not found on disk, the engine falls back to encoding it on the fly using Late Fusion (with PLO linking).

**Stage 2 — Per-Student Interest Vector**

The student's selected interests, application domains, and RDIA priority are each pre-computed domain vectors. These are averaged into a single interest vector.

```
interest_vec = normalize( avg(interest_vecs + app_vecs + rdia_vec) )
```

**Stage 3 — Student Vector**

The competency and interest vectors are blended according to the active `weighting_mode` and normalized:

```
student_vec = normalize( comp_w × competency_vec + int_w × interest_vec )
```

**Stage 4 — Group Vector**

Individual student vectors are aggregated using **element-wise mean pooling**:

```
group_vec = normalize( mean(student_vec_1, student_vec_2, ...) )
```

Mean pooling produces a group vector that reflects the balanced average profile across all members, ensuring no single student dominates the representation.

### 3.2 Project Recommendation Pipeline

The project recommender uses a four-stage hybrid retrieval pipeline.

#### Stage 1: Dense Retrieval

Cosine similarity is computed between the group vector and all pre-loaded project vectors using a single matrix multiplication:

```
scores = project_matrix @ group_vec     # shape: (N,)
```

The top-K projects (K=30) are retained as dense candidates. This is efficient even for thousands of projects and captures semantic relatedness.

#### Stage 2: Sparse Retrieval (BM25)

The group's domain label names (selected interests, applications, RDIA) are used as a keyword query against a BM25 index built from project metadata. BM25 rewards term frequency while penalizing overly common terms, and is particularly effective when the group's selections happen to match exact vocabulary used in a project.

This stage complements dense retrieval by recovering projects that are a strong keyword match but may score lower semantically.

#### Stage 3: Reciprocal Rank Fusion (RRF)

The two ranked lists are merged using RRF:

```
RRF_score(p) = 1/(k + rank_dense(p)) + 1/(k + rank_sparse(p))
```

where k=60 is a smoothing constant. Projects appearing in only one list still receive partial credit. This fusion is ranking-based rather than score-based, making it robust to the different score scales produced by cosine similarity vs. BM25.

#### Stage 4: Policy Re-Ranking

The top candidates from RRF are re-scored using a composite policy formula:

```
Final_Score = α·semantic + β·context + γ·RDIA
```

| Component | Weight | Description |
|---|---|---|
| Semantic (α=0.50) | 50% | Cosine similarity from dense retrieval, shifted to [0,1] |
| Context (β=0.25) | 25% | Fraction of group's application domains matching the project |
| RDIA (γ=0.25) | 25% | Whether the project's RDIA priority matches the group's |

> **Note on tag normalization:** The policy re-ranker normalizes tags (lowercasing and replacing `&` with `and`) before comparing application and RDIA fields, to handle minor inconsistencies in metadata labelling.

### 3.3 Domain Recommendation Logic (Interests, Applications, RDIA)

All three domain recommenders share the same scoring pattern:

```
combined_score = 0.70 × semantic_score + 0.30 × frequency_score
```

- **Semantic score**: Cosine similarity between the group vector and the pre-computed domain vector, shifted from [−1, 1] to [0, 1].
- **Frequency score**: How often the domain appears in the top-10 dense retrieval results, normalized to [0, 1] by dividing by the maximum observed count.

The 70/30 split prioritizes the direct semantic fit of the group's profile to the domain, while giving secondary weight to empirical evidence from past projects.

---

### 4.1 How Outputs Are Explained

Each recommended project includes an `explanation` field generated by the `_explain()` method in `ProjectRecommender`. The explanation is composed from three observable signals:

- **Semantic similarity tier**: Maps the numeric semantic score to a plain-language description ("Highly similar", "Good match", "Partial match")
- **Application domain match**: Lists any application domains shared between the project and the group's selections
- **RDIA alignment**: Notes if the project's RDIA priority matches the group's

Domain outputs include `already_selected` flags that indicate when a recommendation simply confirms a group's existing choice (validation) versus suggesting a new direction (discovery).

---

## 5. Implementation Considerations

### 5.1 Data Flow

```
Phase II (Offline)                     Runtime (Per Request)
─────────────────────────────          ────────────────────────────────────
courses.json                           Group JSON input
    │                                      │
    ▼                                      ▼
[SBERT encode CLOs + PLOs]            EmbeddingEngine.build_group_profile()
    │                                      │
    ▼                                      ├── _build_competency_vec()
embeddings/courses/*.npy                   │     └── load .npy per course
                                           │         (fallback: encode on-the-fly)
projects/*.json                            │         → weighted avg → normalize
    │                                      │
    ▼                                      ├── _build_interest_vec()
[SBERT encode segments]                    │     └── lookup pre-computed domain vecs
    │                                      │         → avg → normalize
    ▼                                      │
embeddings/projects/*.npy                  └── mean_pool(student_vecs) → normalize
embeddings/project_index.json                   = group_vec

Interest/Application/RDIA domains         group_vec, group_meta
    │                                          │
    ▼                                      ┌───┴──────────────────────┐
[SBERT encode domain labels]               ▼                          ▼
    │                                 ProjectRecommender         DomainRecommenders
    ▼                                 (4-stage pipeline)         (semantic + frequency)
engine.interest_vecs
engine.app_vecs
engine.rdia_vecs
```

### 5.2 Key Components

| Component | Description |
|---|---|
| `phase2_embed.py` | One-time offline embedding generation for courses and projects |
| `embedding_engine.py` | Shared engine: model, matrix, BM25, domain vecs, PLO map, group profile builder |
| `recommender_system.py` | Orchestrator: wires engine and recommenders, exposes `recommend_all()` |
| `project_recommender.py` | Dense + sparse retrieval, RRF, policy re-ranking |
| `interest_recommender.py` | Semantic + frequency scoring for all 22 interest domains |
| `application_recommender.py` | Semantic + frequency scoring for all 10 application domains |
| `rdia_recommender.py` | Semantic + frequency scoring for all 4 RDIA priorities |
| `utils.py` | Data loaders, PLO extractor, text extractors, vector math utilities |
| `Summarizer/summarizer.py` | Optional: generates a one-paragraph LLM summary of the top-5 projects via HuggingFace inference |
| `Evaluation/EvaluationMatrices.py` | Offline evaluation script (Precision/Recall/NDCG/MRR) against silver ground truth |

---

## 6. Usage

### 6.1 Setup

```bash
pip install -r requirements.txt
```

### 6.2 Phase II — Generate Embeddings (run once)

```bash
python phase2_embed.py
```

This populates `embeddings/courses/`, `embeddings/projects/`, and `embeddings/project_index.json`.

### 6.3 Running the Recommender

```python
from recommender_system import RecommenderSystem

# Initialize once when server starts
system = RecommenderSystem()

# Call per group request
results = system.recommend_all(group_json)
```

### 6.4 Input Format

```json
{
  "group_id": "G001",
  "weighting_mode": "balanced",
  "students": [
    {
      "student_id": "S001",
      "courses": [
        {"course_code": "CS1465", "grade": "A"},
        {"course_code": "CS1464", "grade": "B+"}
      ],
      "interests":    ["Computer Vision", "AI / ML"],
      "applications": ["Healthcare / Medical"],
      "rdia":         "Health and Wellness"
    }
  ]
}
```

`weighting_mode` is optional and defaults to `"balanced"`. Valid values: `"balanced"`, `"courses_heavy"`, `"interests_heavy"`.

### 6.5 Output Format

```json
{
  "group_id": "G001",
  "group_profile": {
    "selected_interests": [...],
    "selected_applications": [...],
    "selected_rdia": [...]
  },
  "recommended_projects":     [...],
  "recommended_interests":    [...],
  "recommended_applications": [...],
  "recommended_rdia":         [...]
}
```

### 6.6 Running All Test Groups

```bash
python recommender_system.py
```

Reads from `Experimental data sample/Sample groups.json`, runs all groups, prints detailed output, and saves results to `test_results/`.

### 6.7 Evaluation

```bash
cd Evaluation
python EvaluationMatrices.py
```

Compares `test_results/all_results.json` against `Silver Ground Truth/Experimental Groups with Recommendations.json`. On first run, saves a baseline to `evaluation_baseline.json`. Subsequent runs show metric deltas (Δ) against the previous run.

---

## 7. Optional: Summarizer

`Summarizer/summarizer.py` generates a single cohesive paragraph summarizing the top-5 recommended projects, using an LLM via the HuggingFace Inference Router.

**Setup:**

```bash
pip install openai
export HF_TOKEN="your_token_here"
```

**Usage from app:**

```python
from Summarizer.summarizer import generate_summary
summary = generate_summary(top5_projects)
```

The default model is `Qwen/Qwen2.5-1.5B-Instruct` via `featherless-ai`. Override with the `HF_MODEL` environment variable.
