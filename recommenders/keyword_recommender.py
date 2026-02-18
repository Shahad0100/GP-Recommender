"""
recommenders/keyword_recommender.py
-------------------------------------
Keyword Recommender
Responsibility: extract and rank top keywords/themes from the recommended projects.

Scoring = 50% frequency (how often keyword appears in top projects)
        + 50% semantic relevance (cosine sim of keyword embedding vs group vector)
"""

import numpy as np
from collections import Counter
from typing import List


class KeywordRecommender:
    """Extracts and ranks keywords from the top recommended projects."""

    def __init__(self, engine):
        self.engine = engine

    def recommend(
        self,
        top_project_ids: List[str],
        group_vec: np.ndarray,
        top_n: int = 10
    ) -> List[dict]:
        """
        Extract top keywords from the recommended projects and rank them.

        Args:
            top_project_ids : list of project IDs from ProjectRecommender output
            group_vec       : normalized group profile vector (for semantic scoring)
            top_n           : number of keywords to return (default 10)

        Returns:
            List of dicts: keyword, count, coverage, semantic_score, combined_score
        """
        if not top_project_ids:
            return []

        # Count keyword frequency across top recommended projects
        counter = Counter()
        for pid in top_project_ids:
            meta = self.engine.project_index.get(pid, {})
            for kw in meta.get("keywords", []):
                counter[kw.lower().strip()] += 1

        if not counter:
            return []

        total    = len(top_project_ids)
        max_freq = max(counter.values())

        # Take top candidates (buffer before semantic re-ranking)
        candidates = [kw for kw, _ in counter.most_common(top_n * 2)]

        # Encode all candidate keywords in one batch (Late Fusion: single segment each)
        kw_vecs = self.engine.model.encode(
            candidates,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False
        )

        results = []
        for i, kw in enumerate(candidates):
            count      = counter[kw]
            freq_score = count / max_freq                       # normalized [0,1]
            sem_score  = float(np.dot(kw_vecs[i], group_vec))
            sem_norm   = (sem_score + 1.0) / 2.0               # shift to [0,1]
            combined   = 0.50 * freq_score + 0.50 * sem_norm

            results.append({
                "keyword":        kw,
                "count":          count,
                "coverage":       round(count / total, 2),
                "semantic_score": round(sem_norm, 4),
                "combined_score": round(combined, 4),
            })

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[:top_n]
