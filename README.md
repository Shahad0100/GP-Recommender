
### 1. High-Level Architecture

```
                          ┌──────────────────────────────┐
                          │        GROUP JSON INPUT        │
                          │  (courses, grades, interests,  │
                          │   applications, RDIA choice)   │
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
                          │  • build_group_profile()      │
                          └──────────────┬───────────────┘
                                         │  group_vec + group_meta
                        ┌────────────────┼──────────────────┐
                        │                │                  │
                        ▼                ▼                  ▼
              ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
              │   Project    │  │   Interest   │  │ Application  │
              │  Recommender │  │  Recommender │  │  Recommender │
              └──────────────┘  └──────────────┘  └──────────────┘
                        │
              ┌──────────────┐  ┌──────────────┐
              │     RDIA     │  │   Keyword    │
              │  Recommender │  │  Recommender │
              └──────────────┘  └──────────────┘
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
| Course | The knowledge domain of a course, derived from its title, description, and CLO statements |
| Project | The full intellectual content of a past project — its problem, goals, methods, and outcomes |
| Interest domain | The conceptual meaning of a named interest category (e.g., "Computer Vision") |
| Application domain | The focus area of a target field (e.g., "Healthcare / Medical") |
| RDIA priority | The thematic alignment of a Saudi research initiative |
| Student profile | A weighted blend of the student's competency and interest vectors |
| Group profile | A max-pooled aggregation of all individual student vectors |



### 1 The SBERT Model

The system uses `all-MiniLM-L6-v2`.


### 2 Late Fusion Encoding

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
2. Each CLO statement individually

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
| D | 0.45 |
| F | 0.30 |

### 5. Vector Normalization

All vectors — after averaging, weighted averaging, or max-pooling — are L2-normalized. 

---

## 3. Recommendation Logic

### 1. Building the Group Profile


**Stage 1 — Per-Student Competency Vector**

Each course the student has taken contributes a pre-computed embedding vector, weighted by grade. These are combined into a single competency vector using weighted averaging, then normalized.

```
competency_vec = normalize( Σ weight_i × course_vec_i )
```

**Stage 2 — Per-Student Interest Vector**

The student's selected interests, application domains, and RDIA priority are each pre-computed domain vectors. These are averaged into a single interest vector.

```
interest_vec = normalize( avg(interest_vecs + app_vecs + rdia_vec) )
```

**Stage 3 — Student Vector**

The competency and interest vectors are averaged and normalized into one student vector, giving equal weight to academic background and declared preferences.

```
student_vec = normalize( avg(competency_vec, interest_vec) )
```

**Stage 4 — Group Vector**

Individual student vectors are aggregated using **element-wise max pooling**:

```
group_vec = normalize( max(student_vec_1, student_vec_2, ...) )
```

Max pooling ensures the group vector reflects the strongest signal in each dimension across all members. This means that if any student has strong competency in a particular semantic direction, the group profile inherits that strength. This is preferable to averaging — which would dilute individual strengths — in the context of capstone project selection where diverse skills within a group are an asset.

### 3.2 Project Recommendation Pipeline

The project recommender uses a five-stage hybrid retrieval pipeline.

#### Stage 1: Dense Retrieval

Cosine similarity is computed between the group vector and all pre-loaded project vectors using a single matrix multiplication:

```
scores = project_matrix @ group_vec     # shape: (N,)
```

The top-K projects (K=30) are retained as dense candidates. This is efficient even for thousands of projects and captures semantic relatedness.

#### Stage 2: Sparse Retrieval (BM25)

The group's domain label names (selected interests, applications, RDIA) are used as a keyword query against a BM25 index built from project metadata. BM25 rewards term frequency while penalizing overly common terms, and is particularly effective when the group's selections happen to match exact vocabulary used in a project.

This stage complements dense retrieval by recovering projects that are a strong keyword match but may score lower semantically (e.g., a project that uses domain-specific jargon closely aligned with the group's declared interests).

#### Stage 3: Reciprocal Rank Fusion (RRF)

The two ranked lists are merged using RRF:

```
RRF_score(p) = 1/(k + rank_dense(p)) + 1/(k + rank_sparse(p))
```

where k=60 is a smoothing constant. Projects appearing in only one list still receive partial credit. This fusion is ranking-based rather than score-based, making it robust to the different score scales produced by cosine similarity vs. BM25.

#### Stage 4: Policy Re-Ranking

The top candidates from RRF are re-scored using a composite policy formula:

```
Final_Score = α·semantic + β·context + γ·RDIA − ε·diversity_penalty
```

| Component | Weight | Description |
|---|---|---|
| Semantic (α=0.50) | 50% | Cosine similarity from dense retrieval, shifted to [0,1] |
| Context (β=0.25) | 25% | Fraction of group's application domains matching the project |
| RDIA (γ=0.15) | 15% | Whether the project's RDIA priority matches the group's |
| Diversity penalty (ε=0.10) | −10% | Interest domain overlap with already-selected candidates |


#### Stage 5: Maximal Marginal Relevance (MMR)

MMR is applied as a final reranking step to explicitly balance relevance and diversity in the output list:

```
MMR_score = λ · relevance(p, group_vec) − (1−λ) · max_similarity(p, selected)
```

With λ=0.70, the system leans toward relevance while still penalizing projects that are too similar to already-selected ones. Projects are iteratively selected greedily, and each selected project's vector joins the set of "already chosen" vectors used in subsequent iterations.

### 3.3 Domain Recommendation Logic (Interests, Applications, RDIA)

All three domain recommenders share the same scoring pattern:

```
combined_score = 0.70 × semantic_score + 0.30 × frequency_score
```

- **Semantic score**: Cosine similarity between the group vector and the pre-computed domain vector, shifted from [−1, 1] to [0, 1].
- **Frequency score**: How often the domain appears in the top-10 dense retrieval results, normalized to [0, 1] by dividing by the maximum observed count.

The 70/30 split prioritizes the direct semantic fit of the group's profile to the domain, while giving secondary weight to empirical evidence from past projects that the group's profile already resonates with projects in that domain.

### 3. Keyword Recommendation Logic

Keywords are extracted from the top-10 recommended projects and scored using:

```
combined_score = 0.50 × frequency_score + 0.50 × semantic_score
```

Candidate keywords are encoded at query time (one SBERT call per batch), and their vectors are compared against the group vector. This equal split rewards keywords that are both common across recommended projects and semantically central to the group's profile — avoiding noise keywords that appear frequently but are semantically peripheral.

---

### 5.1 How Outputs Are Explained

Each recommended project includes an `explanation` field generated by the `_explain()` method in `ProjectRecommender`. The explanation is composed from three observable signals:

- **Semantic similarity tier**: Maps the numeric semantic score to a plain-language description ("Highly similar", "Good match", "Partial match")
- **Application domain match**: Lists any application domains shared between the project and the group's selections
- **RDIA alignment**: Notes if the project's RDIA priority matches the group's


Domain and keyword recommendations include `already_selected` flags that indicate when a recommendation simply confirms a group's existing choice (validation) versus suggesting a new direction (discovery).

---

## 6. Implementation Considerations

### 6.1 Data Flow

```
Phase II (Offline)                     Runtime (Per Request)
─────────────────────────────          ────────────────────────────────────
courses.json                           Group JSON input
    │                                      │
    ▼                                      ▼
[SBERT encode CLOs]                   EmbeddingEngine.build_group_profile()
    │                                      │
    ▼                                      ├── _build_competency_vec()
embeddings/courses/*.npy                   │     └── load .npy per course
                                           │         → weighted avg → normalize
projects/*.json                            │
    │                                      ├── _build_interest_vec()
    ▼                                      │     └── lookup pre-computed domain vecs
[SBERT encode segments]                    │         → avg → normalize
    │                                      │
    ▼                                      └── max_pool(student_vecs) → normalize
embeddings/projects/*.npy                       = group_vec
embeddings/project_index.json
                                           group_vec, group_meta
Interest/Application/RDIA domains              │
    │                                      ┌───┴──────────────────────┐
    ▼                                      ▼                          ▼
[SBERT encode domain labels]         ProjectRecommender         DomainRecommenders
    │                                (5-stage pipeline)         (semantic + frequency)
    ▼                                      │
engine.interest_vecs                       ▼
engine.app_vecs                      KeywordRecommender
engine.rdia_vecs                     (encode at query time)
```

### 6. Key Components 

| Component | ----- |
|---|---|
| `phase2_embed.py` | One-time offline embedding generation for courses and projects |
| `embedding_engine.py` | Shared engine: model, matrix, BM25, domain vecs, group profile builder |
| `recommender_system.py` | Orchestrator: wires engine and recommenders, exposes `recommend_all()` |
| `project_recommender.py` | Dense + sparse retrieval, RRF, policy re-ranking, MMR |
| `interest_recommender.py` | Semantic + frequency scoring for all 22 interest domains |
| `application_recommender.py` | Semantic + frequency scoring for all 10 application domains |
| `rdia_recommender.py` | Semantic + frequency scoring for all 4 RDIA priorities |
| `keyword_recommender.py` | Keyword extraction, live encoding, frequency + semantic scoring |
| `utils.py` | Data loaders, text extractors, vector math utilities |

---

