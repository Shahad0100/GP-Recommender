"""
recommender_system.py
----------------------
RecommenderSystem — main orchestrator.

Wires EmbeddingEngine + 5 independent recommenders together.
This is the ONLY file your backend needs to import.

Architecture:
    EmbeddingEngine          ← loaded once, shared by all
         │
    ┌────┼────────────────────────────────┐
    ▼    ▼          ▼          ▼          ▼
Project Interest Application RDIA    Keyword
 Rec     Rec      Rec        Rec      Rec

Usage:
    from recommender_system import RecommenderSystem

    # Initialize once when server starts
    system = RecommenderSystem()

    # Call per group request
    results = system.recommend_all(group_json)

    # Or call individual recommenders:
    group_vec, group_meta = system.engine.build_group_profile(group_json)
    projects  = system.project_rec.recommend(group_vec, group_meta)
    interests = system.interest_rec.recommend(group_vec, group_meta)
    apps      = system.app_rec.recommend(group_vec, group_meta)
    rdia      = system.rdia_rec.recommend(group_vec, group_meta)
    keywords  = system.keyword_rec.recommend([p["project_id"] for p in projects], group_vec)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "recommenders"))

from embedding_engine import EmbeddingEngine
from recommenders.project_recommender     import ProjectRecommender
from recommenders.interest_recommender    import InterestRecommender
from recommenders.application_recommender import ApplicationRecommender
from recommenders.rdia_recommender        import RDIARecommender
from recommenders.keyword_recommender     import KeywordRecommender


class RecommenderSystem:
    """
    Top-level orchestrator.
    One shared EmbeddingEngine, 5 independent recommenders.
    Each recommender has a single responsibility.
    """

    def __init__(self):
        print("=" * 55)
        print("  Initializing RecommenderSystem")
        print("=" * 55)

        # Load engine once — all recommenders share it
        self.engine = EmbeddingEngine()

        # Initialize all 5 recommenders with the shared engine
        self.project_rec  = ProjectRecommender(self.engine)
        self.interest_rec = InterestRecommender(self.engine)
        self.app_rec      = ApplicationRecommender(self.engine)
        self.rdia_rec     = RDIARecommender(self.engine)
        self.keyword_rec  = KeywordRecommender(self.engine)

        print("=" * 55)
        print("  RecommenderSystem ready\n")


    def recommend_all(self, group_json: dict) -> dict:
        """
        Run all 5 recommenders for a group. Returns combined output.

        Input group_json:
        {
          "group_id": "G001",
          "students": [
            {
              "student_id": "S001",
              "courses": [
                {"course_code": "CS1465", "grade": 4.5},
                {"course_code": "CS1464", "grade": 3.0}
              ],
              "interests":    ["Computer Vision", "AI / ML"],   // 1-3 items
              "applications": ["Healthcare / Medical"],          // 1-3 items
              "rdia":         "Health and Wellness"             // exactly 1
            }
          ]
        }

        Output:
        {
          "group_id":     str,
          "group_profile": {selected_interests, selected_applications, selected_rdia},

          "recommended_projects":     [...],  // top-10 past projects
          "recommended_interests":    [...],  // top-3 interest domains
          "recommended_applications": [...],  // top-3 application domains
          "recommended_rdia":         [...],  // all 4 RDIA priorities ranked
          "top_keywords":             [...],  // top-10 keywords from top projects
        }
        """
        group_id = group_json.get("group_id", "unknown")
        print(f"[RecommenderSystem] Processing: {group_id}")

        # Build group profile ONCE — shared across all recommenders this request
        group_vec, group_meta = self.engine.build_group_profile(group_json)

        print(f"  Interests   : {group_meta['selected_interests']}")
        print(f"  Applications: {group_meta['selected_applications']}")
        print(f"  RDIA        : {group_meta['selected_rdia']}")

        # Run each recommender independently
        recommended_projects     = self.project_rec.recommend(group_vec, group_meta)
        recommended_interests    = self.interest_rec.recommend(group_vec, group_meta)
        recommended_applications = self.app_rec.recommend(group_vec, group_meta)
        recommended_rdia         = self.rdia_rec.recommend(group_vec, group_meta)

        top_project_ids = [p["project_id"] for p in recommended_projects]
        top_keywords    = self.keyword_rec.recommend(top_project_ids, group_vec)

        print(f"  Done — "
              f"{len(recommended_projects)} projects | "
              f"{len(recommended_interests)} interests | "
              f"{len(recommended_applications)} apps | "
              f"{len(recommended_rdia)} RDIA | "
              f"{len(top_keywords)} keywords\n")

        return {
            "group_id":                group_id,
            "group_profile":           group_meta,
            "recommended_projects":     recommended_projects,
            "recommended_interests":    recommended_interests,
            "recommended_applications": recommended_applications,
            "recommended_rdia":         recommended_rdia,
            "top_keywords":             top_keywords,
        }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — python recommender_system.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    test_group = {
        "group_id": "G001",
        "students": [
            {
                "student_id": "S001",
                "courses": [
                    {"course_code": "CS1465", "grade": 5},
                    {"course_code": "CS1464", "grade": 4},
                    {"course_code": "CS1466", "grade": 4},
                ],
                "interests":    ["Computer Vision", "Artificial Intelligence / Machine Learning"],
                "applications": ["Healthcare / Medical"],
                "rdia":         "Health and Wellness"
            },
            {
                "student_id": "S002",
                "courses": [
                    {"course_code": "CS1462", "grade": 4},
                    {"course_code": "CS1465", "grade": 3},
                ],
                "interests":    ["Natural Language Processing", "Data Science"],
                "applications": ["Healthcare / Medical"],
                "rdia":         "Health and Wellness"
            }
        ]
    }

    system  = RecommenderSystem()
    results = system.recommend_all(test_group)
    sep     = "=" * 55

    print(f"\n{sep}\n  RECOMMENDED PROJECTS\n{sep}")
    for p in results["recommended_projects"]:
        print(f"\n  #{p['rank']} {p['title']}")
        print(f"       Supervisor: {p['supervisor_name']} ({p['academic_year']})")
        print(f"       Domains   : {p['interest']}")
        print(f"       Score     : {p['scores']['final_score']}")
        print(f"       Why       : {p['explanation']}")

    print(f"\n{sep}\n  RECOMMENDED INTERESTS\n{sep}")
    for d in results["recommended_interests"]:
        sel = " ← already selected" if d["already_selected"] else ""
        print(f"  [{d['combined_score']}] {d['name']}{sel}")

    print(f"\n{sep}\n  RECOMMENDED APPLICATIONS\n{sep}")
    for d in results["recommended_applications"]:
        sel = " ← already selected" if d["already_selected"] else ""
        print(f"  [{d['combined_score']}] {d['name']}{sel}")

    print(f"\n{sep}\n  RDIA (all 4 ranked)\n{sep}")
    for r in results["recommended_rdia"]:
        sel = " ← your selection" if r["already_selected"] else ""
        print(f"  [{r['combined_score']}] {r['label']}{sel}")

    print(f"\n{sep}\n  TOP KEYWORDS\n{sep}")
    for k in results["top_keywords"]:
        print(f"  [{k['combined_score']}] {k['keyword']}  "
              f"(in {k['count']} projects, {k['coverage']*100:.0f}% coverage)")
