"""
recommenders/rdia_recommender.py
----------------------------------
RDIA Recommender
Responsibility: rank ALL 4 RDIA priorities by fit to the group's profile.
Returns all 4 ranked (since there are only 4 options total).
"""

import numpy as np
from collections import Counter
from typing import List


class RDIARecommender:
    """Ranks all 4 RDIA priorities by alignment with the group profile."""

    def __init__(self, engine):
        self.engine = engine

    def recommend(
        self,
        group_vec: np.ndarray,
        group_meta: dict
    ) -> List[dict]:
        """
        Score all 4 RDIA priorities and return them ranked (all 4 returned).

        Args:
            group_vec  : normalized group profile vector
            group_meta : {selected_interests, selected_applications, selected_rdia}

        Returns:
            List of 4 dicts: label, description, combined_score, semantic_score,
            frequency_in_top_projects, already_selected
        """
        freq = self._rdia_frequency(group_vec)

        results = []
        for label, rdia_vec in self.engine.rdia_vecs.items():
            sim      = float(np.dot(group_vec, rdia_vec))
            sim_norm = (sim + 1.0) / 2.0
            combined = 0.70 * sim_norm + 0.30 * freq.get(label, 0.0)

            results.append({
                "label":                     label,
                "description":               self.engine.rdia_map.get(label, ""),
                "combined_score":            round(combined, 4),
                "semantic_score":            round(sim_norm, 4),
                "frequency_in_top_projects": freq.get(label, 0),
                "already_selected":          label in group_meta.get("selected_rdia", []),
            })

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results   # all 4 returned

    def _rdia_frequency(self, group_vec: np.ndarray) -> dict:
        """Top-10 dense results → count RDIA frequency → normalize."""
        scores   = self.engine.project_matrix @ group_vec
        top_pids = [self.engine.project_ids[i]
                    for i in np.argsort(scores)[::-1][:10]]

        counter = Counter()
        for pid in top_pids:
            for r in self.engine.project_index.get(pid, {}).get("rdia", []):
                counter[r] += 1

        max_f = max(counter.values()) if counter else 1
        return {k: v / max_f for k, v in counter.items()}
