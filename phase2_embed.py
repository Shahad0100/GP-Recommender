"""
phase2_embed.py
---------------
PHASE II: Embedding Generation using Late Fusion

Run ONCE before using the recommender system.

Late Fusion approach:
  - Each text component is encoded SEPARATELY into its own vector
  - Component vectors are then averaged into one final vector
  - This avoids the 512-token SBERT limit and gives each component equal weight

What gets embedded:
  A. Courses   → one vector per course  (title+desc segment + each CLO separately)
  B. Projects  → one vector per project (10 segments encoded separately, then averaged)
  

Output saved to:
  embeddings/courses/{course_code}.npy
  embeddings/projects/{project_id}.npy
  embeddings/project_index.json   ← metadata for all projects
  

Usage:
  python phase2_embed.py
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

from utils import (
    load_courses,
    load_acm_taxonomy,
    load_all_projects,
    load_plos_from_courses,
    load_interest_domains,
    load_application_domains,
    load_rdia,
    get_course_texts,
    get_project_segments,
    average_vectors,
    normalize,
    save_vector,
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS 
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR         = "data"
PROJECTS_DIR     = os.path.join(DATA_DIR, "projects")
COURSES_PATH     = os.path.join(DATA_DIR, "courses.json")
ACM_PATH         = os.path.join(DATA_DIR, "ACM_CSS_taxonomy.json")

EMBEDDINGS_DIR      = "embeddings"
COURSES_EMB_DIR     = os.path.join(EMBEDDINGS_DIR, "courses")
PROJECTS_EMB_DIR    = os.path.join(EMBEDDINGS_DIR, "projects")


# SBERT model — all-MiniLM-L6-v2 is fast (384-dim), good quality
# Upgrade to all-mpnet-base-v2 (768-dim) for better accuracy at cost of speed
SBERT_MODEL = "all-MiniLM-L6-v2"


# ─────────────────────────────────────────────────────────────────────────────
# LATE FUSION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def encode_late_fusion(model: SentenceTransformer, segments: list) -> np.ndarray:
    """
    Late Fusion: encode each text segment separately, then average the vectors.

    Why Late Fusion (not text concatenation):
      - SBERT has a 512-token limit. Concatenation truncates long texts.
      - Each segment gets its full 512-token budget independently.
      - The averaged vector captures ALL components, not just the first 512 tokens.

    Args:
        model    : loaded SentenceTransformer
        segments : list of text strings (each encoded separately)

    Returns:
        Single normalized vector representing all segments combined.
    """
    if not segments:
        raise ValueError("No segments provided for embedding.")

    # Encode all segments in one batch (faster than one-by-one)
    vectors = model.encode(
        segments,
        normalize_embeddings=True,   # L2-normalize each vector
        batch_size=32,
        show_progress_bar=False
    )

    # Average all component vectors → single representation
    fused = average_vectors(list(vectors))

    # Re-normalize after averaging (averaging can change the magnitude)
    return normalize(fused)


# ─────────────────────────────────────────────────────────────────────────────
# STEP A: Embed Courses
# ─────────────────────────────────────────────────────────────────────────────

def embed_courses(model: SentenceTransformer, course_map: dict, plos_map: dict):
    """
    Generate and save one embedding vector per course using Late Fusion.

    Segments per course:
      - title + description  (1 segment)
      - level 
      - prerequisites
      - credit hours
      - each CLO statement with its PLOs (N segments, one per CLO)
      

    Saved as: embeddings/courses/{course_code}.npy
    """
    print(f"\n[STEP A] Embedding {len(course_map)} courses with PLO context...")
    os.makedirs(COURSES_EMB_DIR, exist_ok=True)

    for code, course in course_map.items():
        segments = get_course_texts(course, plos_map)

        if not segments:
            print(f"  [WARNING] No text found for course {code}, skipping.")
            continue

        vector = encode_late_fusion(model, segments)
        save_vector(vector, os.path.join(COURSES_EMB_DIR, f"{code}.npy"))

    print(f"  ✓ {len(course_map)} course vectors → {COURSES_EMB_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# STEP B: Embed Projects
# ─────────────────────────────────────────────────────────────────────────────

def embed_projects(
    model: SentenceTransformer,
    projects: list,
    acm_map: dict,
    interest_map: dict = None,
    app_map: dict = None,
    rdia_map: dict = None,
) -> dict:
    """
    Generate and save one embedding vector per project using Late Fusion.

    Segments per project (up to 10):
      title | abstract | keywords | problem | aim | objectives |
      results | future_work | domain_labels&des| acm_descriptions

    Each segment encoded separately → averaged → one project vector.

    Also saves project_index.json with all metadata needed at query time.
    Saved as: embeddings/projects/{project_id}.npy
    """
    print(f"\n[STEP B] Embedding {len(projects)} projects...")
    os.makedirs(PROJECTS_EMB_DIR, exist_ok=True)

    project_index = {}
    skipped       = 0

    for project in projects:
        pid = project.get("id", "")
        if not pid:
            print(f"  [WARNING] Project missing 'id', skipping.")
            skipped += 1
            continue

        segments = get_project_segments(project, acm_map, interest_map, app_map, rdia_map)

        if not segments:
            print(f"  [WARNING] No text segments for project {pid}, skipping.")
            skipped += 1
            continue
        
        vector = encode_late_fusion(model, segments)
        save_vector(vector, os.path.join(PROJECTS_EMB_DIR, f"{pid}.npy"))

        # Save metadata for retrieval (NOT embedded — used as structured data)
        clf = project.get("classification", {})
        project_index[pid] = {
            "id":              pid,
            "title":           project.get("title", ""),
            "supervisor_name": project.get("supervisor_name", ""),
            "supervisor_id":   project.get("supervisor_id", ""),
            "academic_year":   project.get("academic_year", ""),
            "semester":        project.get("semester", ""),
            "keywords":        project.get("keywords", []),
            "abstract":        project.get("abstract", ""),
            # Classification fields — used for policy re-ranking
            "application":     clf.get("application", []),
            "interest":        clf.get("interest", []),
            "rdia":            clf.get("rdia", []),
            "acm":             clf.get("acm", []),
        }

    # Save metadata index as JSON
    index_path = os.path.join(EMBEDDINGS_DIR, "project_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(project_index, f, ensure_ascii=False, indent=2)

    print(f"  ✓ {len(project_index)} project vectors → {PROJECTS_EMB_DIR}/")
    print(f"  ✓ project_index.json saved ({skipped} skipped)")
    return project_index


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PHASE II: Embedding Generation (Late Fusion)")
    print("=" * 60)

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # Load reference data
    print("\n[LOADING] Reading data files...")
    course_map   = load_courses(COURSES_PATH)
    plos_map     = load_plos_from_courses(COURSES_PATH)
    acm_map      = load_acm_taxonomy(ACM_PATH)
    interest_map = load_interest_domains(os.path.join(DATA_DIR, "Interest_Domains.json"))
    app_map      = load_application_domains(os.path.join(DATA_DIR, "Application_Domains.json"))
    rdia_map     = load_rdia(os.path.join(DATA_DIR, "RDIA.json"))
    projects     = load_all_projects(PROJECTS_DIR)
    print(f"  Courses : {len(course_map)}")
    print(f"  PLOs    : {len(plos_map)}")
    print(f"  ACM IDs : {len(acm_map)}")
    print(f"  Interests: {len(interest_map)}")
    print(f"  Apps    : {len(app_map)}")
    print(f"  RDIA    : {len(rdia_map)}")
    print(f"  Projects: {len(projects)}")

    # Load SBERT model
    print(f"\n[LOADING] SBERT model: {SBERT_MODEL}")
    print("  First run downloads ~90MB and caches it. Subsequent runs are instant.")
    model = SentenceTransformer(SBERT_MODEL)
    print("  ✓ Model loaded")

    # Run all three embedding steps
    embed_courses(model, course_map, plos_map)
    embed_projects(model, projects, acm_map, interest_map, app_map, rdia_map)
  

    print("\n" + "=" * 60)
    print("  PHASE II COMPLETE")
    print(f"  All vectors saved to: {EMBEDDINGS_DIR}/")
    print("=" * 60)
    


if __name__ == "__main__":
    main()
