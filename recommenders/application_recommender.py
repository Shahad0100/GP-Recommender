"""
recommenders/application_recommender.py
-----------------------------------------
Application Recommender
Responsibility: rank ALL 10 application domains by fit to the group's profile.
Returns top-N recommendations.

Same scoring logic as InterestRecommender but for application domains.
"""

import numpy as np
from collections import Counter
from typing import List


class ApplicationRecommender:
    """Recommends application domains best aligned with the group profile."""

    def __init__(self, engine):
        self.engine = engine

    def recommend(
        self,
        group_vec: np.ndarray,
        group_meta: dict,
        top_n: int = 3
    ) -> List[dict]:
        """
        Score all 10 application domains and return top_n ranked.

        Args:
            group_vec  : normalized group profile vector
            group_meta : {selected_interests, selected_applications, selected_rdia}
            top_n      : number of domains to return (default 3)

        Returns:
            List of dicts: name, description, combined_score, semantic_score,
            frequency_in_top_projects, already_selected
        """
        freq = self._domain_frequency(group_vec, field="application")

        results = []
        for name, domain_vec in self.engine.app_vecs.items():
            sim      = float(np.dot(group_vec, domain_vec))
            sim_norm = (sim + 1.0) / 2.0
            combined = 0.70 * sim_norm + 0.30 * freq.get(name, 0.0)

            results.append({
                "name":                      name,
                "description":               self.engine.app_map.get(name, ""),
                "combined_score":            round(combined, 4),
                "semantic_score":            round(sim_norm, 4),
                "frequency_in_top_projects": freq.get(name, 0),
                "already_selected":          name in group_meta.get("selected_applications", []),
            })

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[:top_n]

    def _domain_frequency(self, group_vec: np.ndarray, field: str) -> dict:
        """Top-10 dense results → count domain frequency → normalize."""
        scores   = self.engine.project_matrix @ group_vec
        top_pids = [self.engine.project_ids[i]
                    for i in np.argsort(scores)[::-1][:10]]

        counter = Counter()
        for pid in top_pids:
            for d in self.engine.project_index.get(pid, {}).get(field, []):
                counter[d] += 1

        max_f = max(counter.values()) if counter else 1
        return {k: v / max_f for k, v in counter.items()}
