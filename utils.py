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
# GRADE → WEIGHT  (grade A+ → F embedding weight)
# ─────────────────────────────────────────────────────────────────────────────

GRADE_WEIGHTS = {
    "A+": 1.00,
    "A": 0.95,
    "B+": 0.85,
    "B": 0.75,
    "C+": 0.65,
    "C": 0.55,
    "D+": 0.45,
    "D": 0.30
}

def grade_to_weight(grade: str) -> float:
    """Convert letter grade to embedding weight."""
    grade = grade.strip().upper()
    return GRADE_WEIGHTS.get(grade, 0.50)

# ─────────────────────────────────────────────────────────────────────────────
# LOADE PLOs from courses.json
# ─────────────────────────────────────────────────────────────────────────────

def load_plos_from_courses(courses_path: str) -> Dict[str, str]:
    """Extract the list of PLOs and return it as a dictionary {plo_id: description}"""

    with open(courses_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    plos_map = {}
    plo_categories = data.get("program_learning_outcomes", {})
    for category in ["knowledge", "skills", "values"]:
        for plo in plo_categories.get(category, []):
            plo_id = plo.get("plo_id")
            plo_desc = plo.get("plo_description")
            if plo_id and plo_desc:
                plos_map[plo_id] = plo_desc
    return plos_map

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_courses(path: str) -> Dict[str, dict]:
    """Load courses.json → {course_code: course_dict}"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Convert list → dict for constant-time lookup
    # Access by course_code becomes O(1) instead of O(n) list search
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
    Load ACM_CSS_taxonomy.json → flat dict {acm_id: 'full_path: description'}

    For each node that has an ID, we store the full ancestral path from the
    top-level category down to the node itself, so the embedding captures the
    complete semantic context (grandparent → parent → child).

    Example for I.2.6:
      "Computing Methodologies > ARTIFICIAL INTELLIGENCE > Learning: Algorithms
       enabling systems to improve performance through experience."
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    flat = {}

    def extract_acm_ids(items, ancestor_names: list):
        """
        Recursively walk the taxonomy tree.
        ancestor_names: ordered list of ancestor node names (root → current parent).
        """
        for item in items:
            name = item.get("name", "")
            desc = item.get("description", "")
            item_id = item.get("id", "")

            # Build the full path label: "Grandparent > Parent > This Node"
            current_path = ancestor_names + ([name] if name else [])

            if item_id:
                path_label = " > ".join(current_path)
                if desc:
                    flat[item_id] = f"{path_label}: {desc}"
                else:
                    flat[item_id] = path_label

            # Recurse into subcategories, passing the updated path
            if item.get("subcategories"):
                extract_acm_ids(item["subcategories"], current_path)

    for category in data["ACM_CSS_taxonomy"]:
        cat_name = category.get("name", "")
        cat_desc = category.get("description", "")
        cat_id   = category.get("id", "")

        if cat_id:
            if cat_desc:
                flat[cat_id] = f"{cat_name}: {cat_desc}"
            else:
                flat[cat_id] = cat_name

        if category.get("subcategories"):
            extract_acm_ids(category["subcategories"], [cat_name] if cat_name else [])

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

def get_course_texts(course: dict, plos_map: Dict[str, str] = None) -> List[str]:
    """
    Extract meaningful text segments from a course.
    Returns: [title+description, level, prerequisites, credit hours, clo1 with Associated plos, clo2 with Associated plos, ...]
    Excludes: course_code, clo_number
    """
    segments = []

    # Title + description as one segment (they describe the same thing)
    title = course.get("course_title", "")
    desc  = course.get("course_description", "")
    if title or desc:
        segments.append(f"{title}. {desc}".strip())
    
    # include other metadata that may affect meaning and retrieval (e.g. prerequisites, credit hours)
    level= course.get("course_level", "")
    if level:
        segments.append(f"Course level: {level}")
    prereq = course.get("prerequisites", "")
    if prereq:
        segments.append(f"Prerequisites: {prereq}")

    credits = course.get("credit_hours", "")
    if credits:
       segments.append(f"Credit hours: {credits}")

    # Each CLO statement with its associated PLO as its own segment
    clos = course.get("course_learning_outcomes", {})
    for category in ["knowledge", "skills", "values"]:
        for clo in clos.get(category, []):
            stmt = clo.get("clo_statement", "").strip()
            
            # Append associated PLO description if available
            if plos_map and stmt:
                mapped_plos = clo.get("mapped_plos", [])
                plo_descriptions = []
                for plo_id in mapped_plos:
                   if plo_id in plos_map:
                       # include both id and description
                       plo_descriptions.append(f"{plo_id}: {plos_map[plo_id]}")
                if plo_descriptions:
                   stmt += " [Associated PLOs: " + " | ".join(plo_descriptions) + "]"
                
        if stmt:
            segments.append(stmt)

    return segments
def get_project_segments(
    project: dict,
    acm_map: Dict[str, str],
    interest_map: Dict[str, str] = None,
    app_map: Dict[str, str] = None,
    rdia_map: Dict[str, str] = None,
) -> List[str]:
    """
    Extract meaningful text segments from a project JSON.
    Each segment is encoded separately (Late Fusion approach).

    Segments:
      1. title
      2. abstract
      3. keywords joined
      4. problem statement + aim
      5. objectives joined
      6. results
      7. future work
      8. domain labels and descriptions (application + interest + rdia joined)
     9. ACM descriptions (codes resolved to text)

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
   
    # problem and aim joined as one segment
    problem_aim = " ".join(filter(None, [intro.get("problem"), intro.get("aim")]))
    if problem_aim:
        segments.append(problem_aim)

    objectives = intro.get("objectives", [])
    if objectives:
        segments.append(" ".join(objectives))

    if conclusion.get("results"):
        segments.append(conclusion["results"])

    if conclusion.get("future_work"):
        segments.append(conclusion["future_work"])

    # Domain labels joined as one segment
    # Domain maps are provided, enrich with descriptions.
    domain_parts = []

    for name in clf.get("interest", []):
        domain_parts.append(f"{name}: {interest_map[name]}")
        

    for name in clf.get("application", []):
        domain_parts.append(f"{name}: {app_map[name]}")
        

    for name in clf.get("rdia", []):
        domain_parts.append(f"{name}: {rdia_map[name]}")
        

    if domain_parts:
        segments.append(" | ".join(domain_parts))

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
    Prevents long-text dilution and improves retrieval accuracy
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
