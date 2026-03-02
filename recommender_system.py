"""
recommender_system.py
----------------------
RecommenderSystem — main orchestrator.

Wires EmbeddingEngine + 4 independent recommenders together.
This is the ONLY file your backend needs to import.

Architecture:
    EmbeddingEngine          ← loaded once, shared by all
         │
    ┌────┼─────────────────────|
    ▼    ▼          ▼          ▼          
Project Interest Application RDIA    
 Rec     Rec      Rec        Rec  

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
"""
import json
import os
import datetime
import sys
from pathlib import Path
    
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "recommenders"))

from embedding_engine import EmbeddingEngine
from recommenders.project_recommender     import ProjectRecommender
from recommenders.interest_recommender    import InterestRecommender
from recommenders.application_recommender import ApplicationRecommender
from recommenders.rdia_recommender        import RDIARecommender



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

        # Initialize all 4 recommenders with the shared engine
        self.project_rec  = ProjectRecommender(self.engine)
        self.interest_rec = InterestRecommender(self.engine)
        self.app_rec      = ApplicationRecommender(self.engine)
        self.rdia_rec     = RDIARecommender(self.engine)
    

        print("=" * 55)
        print("  RecommenderSystem ready\n")


    def recommend_all(self, group_json: dict) -> dict:
        """
        Run all 4 recommenders for a group. Returns combined output.

        Input group_json:
        {
          "group_id": "G001",
          "weighting_mode": "courese_heavy" or "interests_heavy" or defult=balanced
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

        

        print(f"  Done — "
              f"{len(recommended_projects)} projects | "
              f"{len(recommended_interests)} interests | "
              f"{len(recommended_applications)} apps | "
              f"{len(recommended_rdia)} RDIA \n ")
              

    
        return {
            "group_id":                group_id,
            "group_profile":           group_meta,
            "recommended_projects":     recommended_projects,
            "recommended_interests":    recommended_interests,
            "recommended_applications": recommended_applications,
            "recommended_rdia":         recommended_rdia,
            
               }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — python recommender_system.py
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    def load_groups_from_json(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both array format and object with groups property
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "groups" in data:
                return data["groups"]
            else:
                print(f"Unexpected JSON format in {file_path}")
                print("Expected: array of groups or {'groups': [...]}")
                return []
                
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file {file_path}: {e}")
            return []
    
    def print_detailed_results(results):
        """
        Print results in the same detailed format as the original version.
        """
        sep = "=" * 55
        group_id = results.get("group_id", "unknown")
        
        print(f"\n{sep}")
        print(f"  RESULTS FOR GROUP: {group_id}")
        print(f"{sep}")
        
        # Group profile
        profile = results.get("group_profile", {})
        print(f"\nGroup Profile:")
        print(f"  Interests   : {profile.get('selected_interests', [])}")
        print(f"  Applications: {profile.get('selected_applications', [])}")
        print(f"  RDIA        : {profile.get('selected_rdia', 'N/A')}")
        
        # Recommended projects (full details)
        print(f"\n{sep}")
        print(f"  RECOMMENDED PROJECTS")
        print(f"{sep}")
        for p in results["recommended_projects"]:
            print(f"\n  #{p['rank']} {p['title']}")
            print(f"       Supervisor: {p['supervisor_name']} ({p['academic_year']})")
            print(f"       Domains   : {p['interest']}")
            print(f"       Score     : {p['scores']['final_score']}")
            print(f"       Why       : {p['explanation']}")
        
        # Recommended interests
        print(f"\n{sep}")
        print(f"  RECOMMENDED INTERESTS")
        print(f"{sep}")
        for d in results["recommended_interests"]:
            sel = " ← already selected" if d["already_selected"] else ""
            print(f"  [{d['combined_score']}] {d['name']}{sel}")
        
        # Recommended applications
        print(f"\n{sep}")
        print(f"  RECOMMENDED APPLICATIONS")
        print(f"{sep}")
        for d in results["recommended_applications"]:
            sel = " ← already selected" if d["already_selected"] else ""
            print(f"  [{d['combined_score']}] {d['name']}{sel}")
        
        # RDIA priorities
        print(f"\n{sep}")
        print(f"  RDIA (all 4 ranked)")
        print(f"{sep}")
        for r in results["recommended_rdia"]:
            sel = " ← your selection" if r["already_selected"] else ""
            print(f"  [{r['combined_score']}] {r['label']}{sel}")
        
        print()
    
    def save_results_to_file(results, output_dir="test_results"):
        """
        Save test results to a JSON file.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename based on group_id
        group_id = results.get("group_id", "unknown")
        filename = f"results_{group_id}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Save results
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Path to groups JSON file
    BASE_DIR = Path(__file__).resolve().parent
    groups_file = BASE_DIR / "Experimental data sample" / "Sample groups.json" 

    # Check if file exists
    if not os.path.exists(groups_file):
        print(f"File not found: {groups_file}")
        print("Please make sure the file exists and update the 'groups_file' variable.")
        print("Current working directory:", os.getcwd())
        sys.exit(1)
    
    # Load groups from file
    print(f"\n{'='*55}")
    print(f"  LOADING GROUPS FROM: {groups_file}")
    print(f"{'='*55}")
    
    groups = load_groups_from_json(groups_file)
    
    if not groups:
        print("No groups found. Exiting test.")
        sys.exit(1)
    
    print(f"Loaded {len(groups)} groups for testing")
    print(f"Groups: {[g.get('group_id', 'unknown') for g in groups]}\n")
    
    # Initialize the recommender system once
    print("Initializing RecommenderSystem...")
    system = RecommenderSystem()
    
    # Store all results
    all_results = []
    successful_groups = 0
    failed_groups = 0
    
    # Test each group
    print(f"\n{'='*55}")
    print(f"  TESTING ALL GROUPS")
    print(f"{'='*55}")
    
    for i, group in enumerate(groups, 1):
        group_id = group.get('group_id', f'Group_{i}')
        print(f"\n[{i}/{len(groups)}] Processing group: {group_id}")
        print("-" * 55)
        
        try:
            # Get recommendations for this group
            results = system.recommend_all(group)
            
            # Store results
            all_results.append(results)
            successful_groups += 1
            
            # Print detailed results in the original format
            print_detailed_results(results)
            
            # Save individual results to file
            save_results_to_file(results)
            
        except Exception as e:
            failed_groups += 1
            print(f"\nError processing group {group_id}: {e}")
            print("Type of error:", type(e).__name__)
            import traceback
            traceback.print_exc()
            print("\n" + "-" * 55)
    
       # Save all results combined
    if all_results:
        combined_results = {
            "total_groups": len(all_results),
            "successful": successful_groups,
            "failed": failed_groups,
            "timestamp": str(datetime.datetime.now()),
            "results": all_results
        }
        
# Clean the data of any unwanted fields (such as top_keywords)
        for group_result in combined_results["results"]:
            #Remove top_keywords if it exists
            if "top_keywords" in group_result:
                del group_result["top_keywords"]
        
        os.makedirs("test_results", exist_ok=True)
        with open("test_results/all_results.json", 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, ensure_ascii=False, indent=2)
        
        print(f"  Combined results saved to test_results/all_results.json")
    
    # Print final summary
    print(f"\n{'='*55}")
    print(f"  TEST COMPLETED")
    print(f"{'='*55}")
    print(f"Successfully tested: {successful_groups} groups")
    if failed_groups > 0:
        print(f" Failed: {failed_groups} groups")
    print(f"Detailed results printed above for each successful group")
    print(f"Individual results saved in 'test_results/' directory")
    if all_results:
        print(f"Combined results saved in 'test_results/all_results.json'")
