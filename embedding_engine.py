"""
embedding_engine.py
--------------------
Shared Embedding Engine — used by ALL recommenders.

Loaded ONCE at server startup. Never reloaded per request.

Responsibilities:
  1. Load SBERT model
  2. Load all pre-computed project vectors into a matrix (for fast batch search)
  3. Build BM25 index (for sparse keyword retrieval)
  4. Pre-compute domain option vectors (22 interests, 10 apps, 4 RDIA)
  5. Build group profile vector from runtime group JSON

What each recommender gets from here:
  engine.model             → SBERT (for encoding new text at query time)
  engine.project_index     → {pid: metadata_dict}
  engine.project_ids       → ordered list of project IDs
  engine.project_matrix    → (N, D) numpy matrix of all project vectors
  engine.bm25              → BM25 index
  engine.bm25_order        → position → project_id mapping
  engine.interest_map      → {name: description}
  engine.app_map           → {field: focus}
  engine.rdia_map          → {label: description}
  engine.interest_vecs     → {name: np.ndarray}  pre-computed
  engine.app_vecs          → {field: np.ndarray} pre-computed
  engine.rdia_vecs         → {label: np.ndarray} pre-computed
  engine.build_group_profile(group_json) → (group_vec, group_meta)
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from utils import (
    load_courses,
    load_interest_domains,
    load_application_domains,
    load_rdia,
    load_acm_taxonomy,
    get_course_texts,
    encode_late_fusion_engine,
    grade_to_weight,
    weighted_average,
    average_vectors,
    normalize,
    load_vector,
    load_plos_from_courses,
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR       = "data"
EMBEDDINGS_DIR = "embeddings"

COURSES_PATH     = os.path.join(DATA_DIR, "courses.json")
INTEREST_PATH    = os.path.join(DATA_DIR, "Interest_Domains.json")
APPLICATION_PATH = os.path.join(DATA_DIR, "Application_Domains.json")
RDIA_PATH        = os.path.join(DATA_DIR, "RDIA.json")
ACM_PATH         = os.path.join(DATA_DIR, "ACM_CSS_taxonomy.json")

PROJECTS_EMB_DIR      = os.path.join(EMBEDDINGS_DIR, "projects")
COURSES_EMB_DIR       = os.path.join(EMBEDDINGS_DIR, "courses")
PROJECT_INDEX_PATH    = os.path.join(EMBEDDINGS_DIR, "project_index.json")


SBERT_MODEL = "all-MiniLM-L6-v2"


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Shared engine initialized once at startup.
    Passed to every recommender as a dependency.
    """

    def __init__(self):
        print("[EmbeddingEngine] Initializing...")

        # Load reference lookup tables for courses, domains, and PLOs.
        self.course_map   = load_courses(COURSES_PATH)
        # for on‑the‑fly encoding we also need the PLO descriptions
        self.plos_map     = load_plos_from_courses(COURSES_PATH)
        self.interest_map = load_interest_domains(INTEREST_PATH)
        self.app_map      = load_application_domains(APPLICATION_PATH)
        self.rdia_map     = load_rdia(RDIA_PATH)
        self.acm_map      = load_acm_taxonomy(ACM_PATH)

        # Load SBERT model (used for encoding group queries at runtime)
        print(f"  Loading SBERT: {SBERT_MODEL}")
        self.model = SentenceTransformer(SBERT_MODEL)

        # Pre-compute vectors for all domain options (done once, reused every request)
        self._precompute_domain_vecs()

        # Load project metadata and vectors
        with open(PROJECT_INDEX_PATH, "r", encoding="utf-8") as f:
            self.project_index: Dict[str, dict] = json.load(f)


        self._load_project_matrix()
        self._build_bm25()

        print(f"[EmbeddingEngine] Ready — "
              f"{len(self.project_ids)} projects | "
              f"{len(self.interest_vecs)} interests | "
              f"{len(self.app_vecs)} apps | "
              f"{len(self.rdia_vecs)} RDIA\n")


    def _precompute_domain_vecs(self):
        """
        Encode all domain options once at startup using Late Fusion.
        Each domain: [name, description] → 2 segments → averaged vector.
        Stored as dicts for O(1) lookup per domain name.
        """
        # Interest domains (22 options)
        self.interest_vecs: Dict[str, np.ndarray] = {}
        for name, desc in self.interest_map.items():
            self.interest_vecs[name] = self._encode_domain(name, desc)

        # Application domains (10 options)
        self.app_vecs: Dict[str, np.ndarray] = {}
        for field, focus in self.app_map.items():
            self.app_vecs[field] = self._encode_domain(field, focus)

        # RDIA priorities (4 options)
        self.rdia_vecs: Dict[str, np.ndarray] = {}
        for label, desc in self.rdia_map.items():
            self.rdia_vecs[label] = self._encode_domain(label, desc)

        print(f"  Domain vectors: {len(self.interest_vecs)} interests | "
              f"{len(self.app_vecs)} apps | {len(self.rdia_vecs)} RDIA")


    def _encode_domain(self, name: str, description: str) -> np.ndarray:
        """
        Encode a domain option using Late Fusion:
          segment 1: domain name  (short label)
          segment 2: description  (longer explanation)
        Average → normalized vector.
        """
        segments = [name, description] if description else [name]
        vecs = self.model.encode(segments, normalize_embeddings=True,
                                  show_progress_bar=False)
        return normalize(average_vectors(list(vecs)))


    def _load_project_matrix(self):
        """
        Load all pre-computed project .npy files into one numpy matrix.
        Shape: (N, D) — enables fast batch cosine similarity via dot product.
        """
        self.project_ids: List[str] = []
        vecs: List[np.ndarray]      = []

        for pid in self.project_index:
            path = os.path.join(PROJECTS_EMB_DIR, f"{pid}.npy")
            if os.path.exists(path):
                self.project_ids.append(pid)
                vecs.append(load_vector(path))

        if not vecs:
            self.project_matrix = np.empty((0, 384), dtype=np.float32)
        else:
            self.project_matrix = np.array(vecs)
        print(f"  Project matrix loaded: {self.project_matrix.shape}")


    def _build_bm25(self):
        """
        Build BM25 sparse index from project metadata text.
        Only uses keyword-rich fields: title, keywords, interest, application, rdia.
        """
        corpus          : List[List[str]] = []
        self.bm25_order : List[str]       = []

        for pid, meta in self.project_index.items():
            text = " ".join([
                meta.get("title", ""),
                " ".join(meta.get("keywords", [])),
                " ".join(meta.get("interest", [])),
                " ".join(meta.get("application", [])),
                " ".join(meta.get("rdia", [])),
            ]).lower()
            corpus.append(text.split())
            self.bm25_order.append(pid)

        if corpus:
            self.bm25 = BM25Okapi(corpus)
        else:
            self.bm25 = None
        print(f"  BM25 index built: {len(self.bm25_order)} documents")


    # ─────────────────────────────────────────────────────────────────────────
    # GROUP PROFILE BUILDER — called once per request
    # ─────────────────────────────────────────────────────────────────────────

    def build_group_profile(self, group_json: dict) -> Tuple[np.ndarray, dict]:
        """
        Build a single normalized group profile vector from the group JSON.
        Also returns group_meta with all domain selections aggregated.

        For each student:
          competency_vec = weighted_avg(course_vecs, grade_weights)  ← Late Fusion
          interest_vec   = avg(interest_vecs + app_vecs + rdia_vec)  ← pre-computed
          student_vec    = avg(competency_vec, interest_vec) → normalized

        group_vec = max_pool(all student_vecs) → normalized

        Returns:
          group_vec  : np.ndarray
          group_meta : {selected_interests, selected_applications, selected_rdia}
        """
        weighting_mode = group_json.get("weighting_mode", "balanced")
        if weighting_mode == "courses_heavy":
            comp_w, int_w = 0.75, 0.25
        elif weighting_mode == "interests_heavy":
            comp_w, int_w = 0.25, 0.75
        else:
            comp_w, int_w = 0.50, 0.50
            
        student_vecs  = []
        all_interests = set()
        all_apps      = set()
        all_rdia      = set()

        for student in group_json.get("students", []):
            comp_vec = self._build_competency_vec(student)
            int_vec  = self._build_interest_vec(student)

            student_vec = normalize(comp_w * comp_vec + int_w * int_vec)
            student_vecs.append(student_vec)

            all_interests.update(student.get("interests", []))
            all_apps.update(student.get("applications", []))
            rdia = student.get("rdia", "")
            if rdia:
                all_rdia.add(rdia)

        if not student_vecs:
            raise ValueError("Group has no valid students.")
      
        group_vec = np.mean(np.array(student_vecs), axis=0)
        group_vec = normalize(group_vec)
      
        group_meta = {
            "selected_interests":    list(all_interests),
            "selected_applications": list(all_apps),
            "selected_rdia":         list(all_rdia),
        }
        return group_vec, group_meta


    def _build_competency_vec(self, student: dict) -> np.ndarray:
        """
        Competency vector = weighted average of course CLO vectors.
        Uses pre-computed .npy files. Falls back to live encoding if missing.
        Weight per course = grade_to_weight(grade).
        """
        vecs, weights = [], []

        for entry in student.get("courses", []):
            code  = entry.get("course_code", "")
            grade = entry.get("grade", 3)

            path = os.path.join(COURSES_EMB_DIR, f"{code}.npy")
            if os.path.exists(path):
                vec = load_vector(path)
            elif code in self.course_map:
                # Fallback: encode on the fly using Late Fusion (with PLO linking)
                segments = get_course_texts(self.course_map[code], self.plos_map)
                vecs_raw = self.model.encode(segments, normalize_embeddings=True,
                                              show_progress_bar=False)
                vec = normalize(average_vectors(list(vecs_raw)))
            else:
                print(f"  [WARNING] Unknown course '{code}', skipping.")
                continue

            vecs.append(vec)
            weights.append(grade_to_weight(grade))

        if not vecs:
            return np.zeros(self.project_matrix.shape[1])

        return normalize(weighted_average(vecs, weights))


    def _build_interest_vec(self, student: dict) -> np.ndarray:
        """
        Interest vector = average of pre-computed domain vectors for:
          interests (1–3) + applications (1–3) + rdia (exactly 1)
        Uses self.interest_vecs, self.app_vecs, self.rdia_vecs — no encoding needed.
        """
        vecs = []

        for name in student.get("interests", []):
            if name in self.interest_vecs:
                vecs.append(self.interest_vecs[name])

        for name in student.get("applications", []):
            if name in self.app_vecs:
                vecs.append(self.app_vecs[name])

        rdia = student.get("rdia", "")
        if rdia and rdia in self.rdia_vecs:
            vecs.append(self.rdia_vecs[rdia])

        if not vecs:
            return np.zeros(self.project_matrix.shape[1])

        return normalize(average_vectors(vecs))


    # ─────────────────────────────────────────────────────────────────────────
    # CONVENIENCE GETTERS
    # ─────────────────────────────────────────────────────────────────────────

    def get_project_vec(self, pid: str) -> np.ndarray:
        """Load a single project vector by ID."""
        path = os.path.join(PROJECTS_EMB_DIR, f"{pid}.npy")
        return load_vector(path) if os.path.exists(path) else None

