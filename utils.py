"""
utils.py
--------
Data loaders and helper functions shared across all files.
"""

import json
import os
import numpy as np
from typing import List, Dict


# ─────────────────────────────────────────────────────────────────────────────
# GRADE → WEIGHT  (grade 1–5 → embedding weight)
# ─────────────────────────────────────────────────────────────────────────────

GRADE_WEIGHTS = {
    "A+": 1.00,
    "A": 0.95,
    "B+": 0.85,
    "B": 0.75,
    "C+": 0.65,
    "C": 0.55,
    "D": 0.45,
    "F": 0.30
}

def grade_to_weight(grade: str) -> float:
    """Convert letter grade to embedding weight."""
    grade = grade.strip().upper()
    return GRADE_WEIGHTS.get(grade, 0.50)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_courses(path: str) -> Dict[str, dict]:
    """Load courses.json → {course_code: course_dict}"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {c["course_code"]: c for c in data["courses"]}


def load_interest_domains(path: str) -> Dict[str, str]:
    """Load Interest_Domains.json → {name: description}"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["name"]: item["description"] for item in data["DOMAIN_CATEGORIES"]}


def load_application_domains(path: str) -> Dict[str, str]:
    """Load Application_Domains.json → {Field: Focus}"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["Field"]: item["Focus"] for item in data}


def load_rdia(path: str) -> Dict[str, str]:
    """Load RDIA.json → {Label: Description}"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["Label"]: item["Description"] for item in data["RDIA"]}


def load_acm_taxonomy(path: str) -> Dict[str, str]:
    """
    Load ACM_CSS_taxonomy.json → flat dict {acm_id: 'name: description'}
    Converts ACM codes like 'I.2.6' to human-readable descriptions
    so they can be embedded meaningfully.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    flat = {}
    for category in data["ACM_CSS_taxonomy"]:
        for sub in category.get("subcategories", []):
            acm_id = sub.get("id", "")
            if acm_id:
                flat[acm_id] = f"{sub.get('name','')}: {sub.get('description','')}"
    return flat


def load_all_projects(projects_dir: str) -> List[dict]:
    """Load all project JSON files from a folder."""
    projects = []
    for fname in sorted(os.listdir(projects_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(projects_dir, fname), "r", encoding="utf-8") as f:
                try:
                    projects.append(json.load(f))
                except json.JSONDecodeError as e:
                    print(f"  [WARNING] Skipping {fname}: {e}")
    return projects


# ─────────────────────────────────────────────────────────────────────────────
# TEXT EXTRACTORS
# Each function returns a list of text segments (NOT one big string).
# Phase II encodes each segment separately then averages (Late Fusion).
# ─────────────────────────────────────────────────────────────────────────────

def get_course_texts(course: dict) -> List[str]:
    """
    Extract meaningful text segments from a course.
    Returns: [title+description, clo1, clo2, ...]
    Excludes: course_code, credit_hours, clo_number, mapped_plos
    """
    segments = []

    # Title + description as one segment (they describe the same thing)
    title = course.get("course_title", "")
    desc  = course.get("course_description", "")
    if title or desc:
        segments.append(f"{title}. {desc}".strip())

    # Each CLO statement as its own segment
    clos = course.get("course_learning_outcomes", {})
    for category in ["knowledge", "skills", "values"]:
        for clo in clos.get(category, []):
            stmt = clo.get("clo_statement", "").strip()
            if stmt:
                segments.append(stmt)

    return segments


def get_project_segments(project: dict, acm_map: Dict[str, str]) -> List[str]:
    """
    Extract meaningful text segments from a project JSON.
    Each segment is encoded separately (Late Fusion approach).

    Segments:
      1. title
      2. abstract
      3. keywords joined
      4. problem statement
      5. aim
      6. objectives joined
      7. results
      8. future work
      9. domain labels (application + interest + rdia joined)
     10. ACM descriptions (codes resolved to text)

    Excludes: id, supervisor_name, supervisor_id, academic_year, semester, acm codes directly
    """
    intro      = project.get("introduction", {})
    conclusion = project.get("conclusion", {})
    clf        = project.get("classification", {})

    # Resolve ACM codes to human-readable text
    acm_texts = [acm_map.get(code, "") for code in clf.get("acm", []) if acm_map.get(code)]

    segments = []

    # Each major field as its own segment
    if project.get("title"):
        segments.append(project["title"])

    if project.get("abstract"):
        segments.append(project["abstract"])

    keywords = project.get("keywords", [])
    if keywords:
        segments.append(" ".join(keywords))

    if intro.get("problem"):
        segments.append(intro["problem"])

    if intro.get("aim"):
        segments.append(intro["aim"])

    objectives = intro.get("objectives", [])
    if objectives:
        segments.append(" ".join(objectives))

    if conclusion.get("results"):
        segments.append(conclusion["results"])

    if conclusion.get("future_work"):
        segments.append(conclusion["future_work"])

    # Domain labels joined as one segment
    domain_labels = (
        clf.get("application", []) +
        clf.get("interest", []) +
        clf.get("rdia", [])
    )
    if domain_labels:
        segments.append(" ".join(domain_labels))

    # ACM descriptions as one segment
    if acm_texts:
        segments.append(" ".join(acm_texts))

    return [s.strip() for s in segments if s.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# VECTOR MATH
# ─────────────────────────────────────────────────────────────────────────────

def weighted_average(vectors: List[np.ndarray], weights: List[float]) -> np.ndarray:
    """Weighted average of vectors. Weights are normalized internally."""
    vecs = np.array(vectors)
    w    = np.array(weights, dtype=float)
    w    = w / w.sum()
    return np.average(vecs, axis=0, weights=w)


def average_vectors(vectors: List[np.ndarray]) -> np.ndarray:
    """Simple unweighted mean of vectors."""
    return np.mean(np.array(vectors), axis=0)


def normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector. Returns zero vector if norm is zero."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def save_vector(vec: np.ndarray, path: str):
    """Save numpy vector to .npy file."""
    np.save(path, vec)


def load_vector(path: str) -> np.ndarray:
    """Load numpy vector from .npy file."""
    return np.load(path)


def encode_late_fusion_engine(model, segments: list) -> np.ndarray:
    """
    Late Fusion encoding for use inside EmbeddingEngine at query time.
    Encodes each text segment separately, then averages the vectors.
    """
    if not segments:
        raise ValueError("No segments provided.")
    vectors = model.encode(
        segments,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False
    )
    return normalize(average_vectors(list(vectors)))
