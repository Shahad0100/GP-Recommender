"""
recommenders/interest_recommender.py
--------------------------------------
Interest Recommender
Responsibility: rank ALL 22 interest domains by fit to the group's profile.
Returns top-N recommendations.

Scoring = 70% semantic similarity + 30% frequency in top retrieved projects
"""

import numpy as np
from collections import Counter
from typing import List


class InterestRecommender:
    """Recommends interest domains best aligned with the group profile."""

    def __init__(self, engine):
        self.engine = engine

    def recommend(
        self,
        group_vec: np.ndarray,
        group_meta: dict,
        top_n: int = 3
    ) -> List[dict]:
        """
        Score all 22 interest domains and return top_n ranked.

        Scoring combines:
          70% semantic similarity  → cosine sim(group_vec, domain_vec)
          30% frequency evidence   → how often domain appears in top-10 projects

        Args:
            group_vec  : normalized group profile vector
            group_meta : {selected_interests, selected_applications, selected_rdia}
            top_n      : number of domains to return (default 3)

        Returns:
            List of dicts: name, description, combined_score, semantic_score,
            frequency_in_top_projects, already_selected
        """
        freq = self._domain_frequency(group_vec, field="interest")

        results = []
        for name, domain_vec in self.engine.interest_vecs.items():
            # Cosine similarity (vectors are L2-normalized)
            sim      = float(np.dot(group_vec, domain_vec))
            sim_norm = (sim + 1.0) / 2.0   # shift [-1,1] → [0,1]
            combined = 0.70 * sim_norm + 0.30 * freq.get(name, 0.0)

            results.append({
                "name":                      name,
                "description":               self.engine.interest_map.get(name, ""),
                "combined_score":            round(combined, 4),
                "semantic_score":            round(sim_norm, 4),
                "frequency_in_top_projects": freq.get(name, 0),
                "already_selected":          name in group_meta.get("selected_interests", []),
            })

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[:top_n]

    def _domain_frequency(self, group_vec: np.ndarray, field: str) -> dict:
        """
        Get top-10 dense results, count domain frequency, normalize to [0,1].
        """
        scores   = self.engine.project_matrix @ group_vec
        top_pids = [self.engine.project_ids[i]
                    for i in np.argsort(scores)[::-1][:10]]

        counter = Counter()
        for pid in top_pids:
            for d in self.engine.project_index.get(pid, {}).get(field, []):
                counter[d] += 1

        max_f = max(counter.values()) if counter else 1
        return {k: v / max_f for k, v in counter.items()}
