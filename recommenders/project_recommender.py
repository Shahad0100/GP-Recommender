"""
recommenders/project_recommender.py
-------------------------------------
Project Recommender
Responsibility: find the top-N most relevant PAST PROJECTS for a group.

Pipeline:
  1. Dense retrieval   → cosine similarity (group_vec vs all project vectors)
  2. Sparse retrieval  → BM25 keyword matching
  3. RRF fusion        → combine both ranked lists
  4. Policy re-ranking → α·semantic + β·context + γ·RDIA - ε·diversity
  5. MMR               → balance relevance vs diversity in final output
"""

import numpy as np
from typing import List, Dict, Tuple

# ── Hyperparameters ───────────────────────────────────────────────────────────
TOP_K_DENSE  = 30    # candidates from dense search
TOP_K_SPARSE = 30    # candidates from BM25 search
TOP_N_RERANK = 15    # candidates entering policy re-ranking
TOP_FINAL    = 10    # final recommendations returned
RRF_K        = 60    # RRF constant (standard value)

# Final score weights — tune during evaluation
ALPHA   = 0.50   # semantic similarity
BETA    = 0.25   # application domain alignment
GAMMA   = 0.15   # RDIA alignment
EPSILON = 0.10   # diversity penalty

MMR_LAMBDA = 0.70   # higher → more relevant, lower → more diverse


class ProjectRecommender:
    """
    Recommends similar past projects using a hybrid pipeline.
    Receives EmbeddingEngine as dependency — loads nothing itself.
    """

    def __init__(self, engine):
        self.engine = engine

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────────────────

    def recommend(
        self,
        group_vec: np.ndarray,
        group_meta: dict,
        top_final: int = TOP_FINAL
    ) -> List[dict]:
        """
        Run full pipeline and return ranked project recommendations.

        Args:
            group_vec  : normalized group profile vector from EmbeddingEngine
            group_meta : {selected_interests, selected_applications, selected_rdia}
            top_final  : number of final recommendations

        Returns:
            List of project dicts with rank, metadata, scores, explanation.
        """
        dense          = self._dense_retrieval(group_vec)
        sparse         = self._sparse_retrieval(group_meta)
        dense_score_map = {pid: s for pid, s in dense}

        fused    = self._rrf_fusion(dense, sparse)
        reranked = self._policy_rerank(fused, group_meta, dense_score_map)
        final    = self._mmr(reranked, group_vec, top_final)

        return [self._format(rank, item, group_meta)
                for rank, item in enumerate(final, 1)]

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Dense Retrieval
    # ─────────────────────────────────────────────────────────────────────────

    def _dense_retrieval(self, query_vec: np.ndarray) -> List[Tuple[str, float]]:
        """
        Batch dot product: group_vec · project_matrix.T
        Vectors are L2-normalized → dot product = cosine similarity.
        Returns top-K (project_id, score) sorted descending.
        """
        scores  = self.engine.project_matrix @ query_vec
        top_idx = np.argsort(scores)[::-1][:TOP_K_DENSE]
        return [(self.engine.project_ids[i], float(scores[i])) for i in top_idx]

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Sparse Retrieval (BM25)
    # ─────────────────────────────────────────────────────────────────────────

    def _sparse_retrieval(self, group_meta: dict) -> List[Tuple[str, float]]:
        """
        BM25 keyword search. Query = group's domain label names.
        Returns only results with non-zero BM25 scores.
        """
        terms = (
            group_meta["selected_interests"] +
            group_meta["selected_applications"] +
            group_meta["selected_rdia"]
        )
        if not terms:
            return []

        tokens  = " ".join(terms).lower().split()
        scores  = self.engine.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:TOP_K_SPARSE]
        return [
            (self.engine.bm25_order[i], float(scores[i]))
            for i in top_idx if scores[i] > 0
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: RRF Fusion
    # ─────────────────────────────────────────────────────────────────────────

    def _rrf_fusion(
        self,
        dense: List[Tuple[str, float]],
        sparse: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        RRF_score(d) = 1/(k + rank_dense) + 1/(k + rank_sparse)
        Projects appearing in only one list still get partial credit.
        """
        scores: Dict[str, float] = {}
        for rank, (pid, _) in enumerate(dense, 1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (RRF_K + rank)
        for rank, (pid, _) in enumerate(sparse, 1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (RRF_K + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Policy Re-Ranking
    # ─────────────────────────────────────────────────────────────────────────

    def _policy_rerank(
        self,
        fused: List[Tuple[str, float]],
        group_meta: dict,
        dense_scores: Dict[str, float]
    ) -> List[dict]:
        """
        Final_Score = α·semantic + β·context_alignment + γ·RDIA - ε·diversity
        """
        candidates      = fused[:TOP_N_RERANK * 2]
        selected_so_far = []
        scored          = []

        for pid, rrf_score in candidates:
            if pid not in self.engine.project_index:
                continue
            meta = self.engine.project_index[pid]

            # Semantic: shift [-1,1] to [0,1]
            sem = (dense_scores.get(pid, 0.0) + 1.0) / 2.0

            # Context alignment: fraction of group apps matched
            proj_apps  = set(a.lower() for a in meta.get("application", []))
            group_apps = set(a.lower() for a in group_meta["selected_applications"])
            ctx = len(proj_apps & group_apps) / len(group_apps) if group_apps else 0.0

            # RDIA alignment
            def n(s): return s.lower().replace("&", "and").strip()
            proj_rdia  = set(n(r) for r in meta.get("rdia", []))
            group_rdia = set(n(r) for r in group_meta["selected_rdia"])
            rdia = len(proj_rdia & group_rdia) / len(group_rdia) if group_rdia else 0.0

            # Diversity penalty: interest domain overlap with already-selected
            proj_int = set(meta.get("interest", []))
            if selected_so_far:
                overlaps = [
                    len(proj_int & set(s.get("interest", [])))
                    for s in selected_so_far
                ]
                div = min(np.mean(overlaps) / 3.0, 1.0)
            else:
                div = 0.0

            final = ALPHA * sem + BETA * ctx + GAMMA * rdia - EPSILON * div

            scored.append({
                "id": pid, "meta": meta, "rrf_score": rrf_score,
                "final_score": final, "semantic_sim": sem,
                "context_score": ctx, "rdia_score": rdia,
                "diversity_pen": div,
            })
            selected_so_far.append(meta)

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        return scored[:TOP_N_RERANK]

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: MMR
    # ─────────────────────────────────────────────────────────────────────────

    def _mmr(
        self,
        candidates: List[dict],
        query_vec: np.ndarray,
        top_final: int
    ) -> List[dict]:
        """
        MMR = λ·relevance - (1-λ)·max_similarity_to_already_selected
        Iteratively picks best next project balancing relevance + diversity.
        """
        selected, remaining, selected_vecs = [], list(candidates), []

        while remaining and len(selected) < top_final:
            best_score, best_idx = float("-inf"), 0

            for i, cand in enumerate(remaining):
                vec = self.engine.get_project_vec(cand["id"])
                if vec is None:
                    continue
                relevance = float(np.dot(vec, query_vec))
                max_sim   = max(
                    (float(np.dot(vec, sv)) for sv in selected_vecs), default=0.0
                )
                mmr = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * max_sim
                if mmr > best_score:
                    best_score, best_idx = mmr, i

            chosen = remaining.pop(best_idx)
            vec    = self.engine.get_project_vec(chosen["id"])
            if vec is not None:
                selected_vecs.append(vec)
            selected.append(chosen)

        return selected

    # ─────────────────────────────────────────────────────────────────────────
    # FORMAT + EXPLAIN
    # ─────────────────────────────────────────────────────────────────────────

    def _format(self, rank: int, item: dict, group_meta: dict) -> dict:
        meta = item["meta"]
        return {
            "rank":            rank,
            "project_id":      item["id"],
            "title":           meta.get("title", ""),
            "supervisor_name": meta.get("supervisor_name", ""),
            "supervisor_id":   meta.get("supervisor_id", ""),
            "academic_year":   meta.get("academic_year", ""),
            "keywords":        meta.get("keywords", []),
            "application":     meta.get("application", []),
            "interest":        meta.get("interest", []),
            "rdia":            meta.get("rdia", []),
            "scores": {
                "final_score":   round(item["final_score"], 4),
                "semantic_sim":  round(item["semantic_sim"], 4),
                "context_score": round(item["context_score"], 4),
                "rdia_score":    round(item["rdia_score"], 4),
                "rrf_score":     round(item["rrf_score"], 6),
            },
            "explanation": self._explain(item, group_meta),
        }

    def _explain(self, item: dict, group_meta: dict) -> str:
        meta, parts = item["meta"], []
        sem = item["semantic_sim"]
        if   sem > 0.75: parts.append("Highly similar to the group's profile.")
        elif sem > 0.55: parts.append("Good match with the group's background.")
        else:            parts.append("Partial match with the group profile.")

        matched = (
            set(a.lower() for a in meta.get("application", [])) &
            set(a.lower() for a in group_meta["selected_applications"])
        )
        if matched:
            parts.append(f"Matches application domain(s): {', '.join(matched)}.")

        def n(s): return s.lower().replace("&", "and").strip()
        if (set(n(r) for r in meta.get("rdia", [])) &
                set(n(r) for r in group_meta["selected_rdia"])):
            parts.append("Aligns with the group's RDIA priority.")

        return " ".join(parts) if parts else "Selected based on overall profile similarity."
