import os
import json
import numpy as np
from collections import defaultdict
from datetime import datetime
import glob

# ------------------------------------------------------------
# PATHS - ADAPTED TO YOUR FOLDER STRUCTURE
# ------------------------------------------------------------

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to reach the project root
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Input files are in the same folder as the script
SILVER_GT_PATH = os.path.join(SCRIPT_DIR, "Experimental Groups with Recommendations.json")
MODEL_RESULTS_PATH = os.path.join(SCRIPT_DIR, "all_results.json")

# Projects folder is in data directory at project root
PROJECTS_FOLDER_PATH = os.path.join(PROJECT_ROOT, "data", "projects")

# Output files will be saved in the same folder as the script
BASELINE_PATH = os.path.join(SCRIPT_DIR, "evaluation_baseline.json")
HISTORY_PATH = os.path.join(SCRIPT_DIR, "evaluation_history.json")

# Print paths for debugging
print("=" * 70)
print("PATH CONFIGURATION")
print("=" * 70)
print(f"Script location: {SCRIPT_DIR}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Silver GT path: {SILVER_GT_PATH}")
print(f"Model results path: {MODEL_RESULTS_PATH}")
print(f"Projects folder: {PROJECTS_FOLDER_PATH}")
print(f"Baseline save path: {BASELINE_PATH}")
print(f"History save path: {HISTORY_PATH}")
print("=" * 70)

# Different K values for evaluation
K_VALUES_PROJECTS = [3, 5, 7, 10]  # For projects
K_VALUES_INTEREST_APP = [1, 3]      # For interests and applications

# ------------------------------------------------------------
# BASIC METRICS
# ------------------------------------------------------------

def precision_at_k(rec, rel, k):
    rec_k = rec[:k]
    return len(set(rec_k) & set(rel)) / k if k else 0

def recall_at_k(rec, rel, k):
    rec_k = rec[:k]
    return len(set(rec_k) & set(rel)) / len(rel) if rel else 0

def mrr(rec, rel):
    for i, r in enumerate(rec, 1):
        if r in rel:
            return 1 / i
    return 0

def dcg_at_k(rec, rel, k):
    score = 0
    for i, r in enumerate(rec[:k], 1):
        if r in rel:
            score += 1 / np.log2(i + 1)
    return score

def ndcg_at_k(rec, rel, k):
    dcg = dcg_at_k(rec, rel, k)
    ideal = dcg_at_k(rel[:k], rel, k)
    return dcg / ideal if ideal > 0 else 0

def accuracy(rec, rel):
    if not rec or not rel:
        return 0
    correct = len(set(rec) & set(rel))
    return correct / len(rec) if rec else 0

def f1_score(rec, rel):
    if not rec or not rel:
        return 0
    rec_set = set(rec)
    rel_set = set(rel)

    precision = len(rec_set & rel_set) / len(rec_set) if rec_set else 0
    recall = len(rec_set & rel_set) / len(rel_set) if rel_set else 0

    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

# ------------------------------------------------------------
# Catalog Coverage - based on files in the projects folder
# ------------------------------------------------------------

def get_all_projects_from_folder():
    if not os.path.exists(PROJECTS_FOLDER_PATH):
        print(f"\nERROR: Projects folder not found at: {PROJECTS_FOLDER_PATH}")
        print("Looking for projects in alternative location...")
        
        # Try alternative path: same folder as script
        alt_path = os.path.join(SCRIPT_DIR, "projects")
        if os.path.exists(alt_path):
            print(f"Found projects at: {alt_path}")
            return get_projects_from_path(alt_path)
        else:
            print(f"Projects folder not found at {alt_path} either")
            return set()
    
    return get_projects_from_path(PROJECTS_FOLDER_PATH)

def get_projects_from_path(folder_path):
    project_files = glob.glob(os.path.join(folder_path, "*.json"))
    project_ids = []

    for file_path in project_files:
        file_name = os.path.basename(file_path)
        project_id = file_name.replace('.json', '')
        project_ids.append(project_id)

    print(f"Found {len(project_ids)} project files in {folder_path}")
    return set(project_ids)

def catalog_coverage(recommended_projects, all_projects_set):
    unique_recommended = set()

    for group_recs in recommended_projects:
        for rec in group_recs[:K_VALUES_PROJECTS[-1]]:
            unique_recommended.add(rec["project_id"])

    intersection = unique_recommended & all_projects_set
    coverage = len(intersection) / len(all_projects_set) if all_projects_set else 0

    return coverage, unique_recommended, intersection

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

def load_silver_gt():
    if not os.path.exists(SILVER_GT_PATH):
        raise FileNotFoundError(f"Silver Ground Truth file not found at: {SILVER_GT_PATH}")
    
    with open(SILVER_GT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    silver_dict = {}
    for group in data:
        group_id = group["group_id"]
        silver_dict[group_id] = {
            "recommended_projects": group.get("recommended_projects", []),
            "top_interests": group.get("top_interests", []),
            "top_applications": group.get("top_applications", []),
            "rdia_category": group.get("rdia_category", None)
        }
    return silver_dict

def load_model_results():
    if not os.path.exists(MODEL_RESULTS_PATH):
        raise FileNotFoundError(f"Model results file not found at: {MODEL_RESULTS_PATH}")
    
    with open(MODEL_RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results_dict = {}
    for group in data["results"]:
        group_id = group["group_id"]
        results_dict[group_id] = {
            "recommended_projects": group.get("recommended_projects", []),
            "recommended_interests": group.get("recommended_interests", []),
            "recommended_applications": group.get("recommended_applications", []),
            "recommended_rdia": group.get("recommended_rdia", []),
            "group_profile": group.get("group_profile", {})
        }
    return results_dict

# ------------------------------------------------------------
# EVALUATE
# ------------------------------------------------------------

def evaluate():
    print("\nLoading data...")
    
    # Check if required files exist in script directory
    if not os.path.exists(SILVER_GT_PATH):
        print(f"\nError: Silver Ground Truth file not found at: {SILVER_GT_PATH}")
        print("Please place 'Experimental Groups with Recommendations.json' in the same folder as this script.")
        return
    
    if not os.path.exists(MODEL_RESULTS_PATH):
        print(f"\nError: Model results file not found at: {MODEL_RESULTS_PATH}")
        print("Please place 'all_results.json' in the same folder as this script.")
        return
    
    try:
        silver = load_silver_gt()
        results = load_model_results()
    except Exception as e:
        print(f"\nError loading files: {e}")
        return

    print(f"Loaded {len(silver)} groups from silver GT")
    print(f"Loaded {len(results)} groups from model results")

    all_projects_set = get_all_projects_from_folder()
    if len(all_projects_set) == 0:
        print("\nWarning: No project files found. Catalog coverage will be 0.")
    
    print(f"Total projects in folder: {len(all_projects_set)}")

    project_metrics_by_k = {k: defaultdict(list) for k in K_VALUES_PROJECTS}
    interest_metrics_by_k = {k: defaultdict(list) for k in K_VALUES_INTEREST_APP}
    application_metrics_by_k = {k: defaultdict(list) for k in K_VALUES_INTEREST_APP}
    rdia_metrics = defaultdict(list)

    all_recommendations = []

    for gid, gt in silver.items():
        if gid not in results:
            print(f"Warning: group {gid} not in model results, skipping.")
            continue

        # Ground truth data (from Silver Ground Truth)
        gt_projects = [p["project_id"] for p in gt.get("recommended_projects", [])]
        gt_interests = gt.get("top_interests", [])
        gt_applications = gt.get("top_applications", [])
        gt_rdia_raw = gt.get("rdia_category")
        gt_rdia = [gt_rdia_raw] if gt_rdia_raw else []

        # Model recommendations (from all_results.json)
        model_group = results[gid]
        rec_projects = model_group.get("recommended_projects", [])
        rec_ids = [p["project_id"] for p in rec_projects]
        all_recommendations.append(rec_projects)

        rec_interests = [i["name"] for i in model_group.get("recommended_interests", [])[:K_VALUES_INTEREST_APP[-1]]]
        rec_applications = [a["name"] for a in model_group.get("recommended_applications", [])[:K_VALUES_INTEREST_APP[-1]]]
        rec_rdia = [r["label"] for r in model_group.get("recommended_rdia", [])]

        # Project Metrics
        for k in K_VALUES_PROJECTS:
            project_metrics_by_k[k]["precision"].append(precision_at_k(rec_ids, gt_projects, k))
            project_metrics_by_k[k]["recall"].append(recall_at_k(rec_ids, gt_projects, k))
            project_metrics_by_k[k]["ndcg"].append(ndcg_at_k(rec_ids, gt_projects, k))

        project_metrics_by_k[K_VALUES_PROJECTS[0]]["mrr"].append(mrr(rec_ids, gt_projects))

        # Interest Metrics
        if gt_interests:
            for k in K_VALUES_INTEREST_APP:
                interest_metrics_by_k[k]["precision"].append(precision_at_k(rec_interests, gt_interests, k))
                interest_metrics_by_k[k]["recall"].append(recall_at_k(rec_interests, gt_interests, k))
                interest_metrics_by_k[k]["ndcg"].append(ndcg_at_k(rec_interests, gt_interests, k))
            interest_metrics_by_k[K_VALUES_INTEREST_APP[0]]["mrr"].append(mrr(rec_interests, gt_interests))

        # Application Metrics
        if gt_applications:
            for k in K_VALUES_INTEREST_APP:
                application_metrics_by_k[k]["precision"].append(precision_at_k(rec_applications, gt_applications, k))
                application_metrics_by_k[k]["recall"].append(recall_at_k(rec_applications, gt_applications, k))
                application_metrics_by_k[k]["ndcg"].append(ndcg_at_k(rec_applications, gt_applications, k))
            application_metrics_by_k[K_VALUES_INTEREST_APP[0]]["mrr"].append(mrr(rec_applications, gt_applications))

        # RDIA Metrics
        if gt_rdia:
            gt_value = gt_rdia[0]
            total_recs = len(rec_rdia)
            
            hit = 1 if gt_value in rec_rdia else 0
            rdia_metrics["hit_rate"].append(hit)
            
            if hit:
                rank = rec_rdia.index(gt_value) + 1
                rdia_metrics["rank"].append(rank)
                rdia_metrics["mrr"].append(1.0 / rank)
            else:
                rdia_metrics["rank"].append(total_recs + 1)  
                rdia_metrics["mrr"].append(0)

    # ------------------------------------------------------------
    # CALCULATE AVERAGES
    # ------------------------------------------------------------

    # Project metrics
    avg_project_by_k = {}
    for k in K_VALUES_PROJECTS:
        avg_project_by_k[k] = {}
        for metric, values in project_metrics_by_k[k].items():
            if values:
                avg_project_by_k[k][metric] = float(np.mean(values))

    mrr_value = float(np.mean(project_metrics_by_k[K_VALUES_PROJECTS[0]]["mrr"])) if project_metrics_by_k[K_VALUES_PROJECTS[0]]["mrr"] else 0
    for k in K_VALUES_PROJECTS:
        avg_project_by_k[k]["mrr"] = mrr_value

    # Interest metrics
    avg_interest_by_k = {}
    for k in K_VALUES_INTEREST_APP:
        avg_interest_by_k[k] = {}
        for metric, values in interest_metrics_by_k[k].items():
            if values:
                avg_interest_by_k[k][metric] = float(np.mean(values))

    mrr_interest = float(np.mean(interest_metrics_by_k[K_VALUES_INTEREST_APP[0]]["mrr"])) if interest_metrics_by_k[K_VALUES_INTEREST_APP[0]]["mrr"] else 0
    for k in K_VALUES_INTEREST_APP:
        avg_interest_by_k[k]["mrr"] = mrr_interest

    # Application metrics
    avg_application_by_k = {}
    for k in K_VALUES_INTEREST_APP:
        avg_application_by_k[k] = {}
        for metric, values in application_metrics_by_k[k].items():
            if values:
                avg_application_by_k[k][metric] = float(np.mean(values))

    mrr_application = float(np.mean(application_metrics_by_k[K_VALUES_INTEREST_APP[0]]["mrr"])) if application_metrics_by_k[K_VALUES_INTEREST_APP[0]]["mrr"] else 0
    for k in K_VALUES_INTEREST_APP:
        avg_application_by_k[k]["mrr"] = mrr_application

    # Catalog coverage
    coverage, unique_recommended, intersection = catalog_coverage(all_recommendations, all_projects_set)

    # RDIA metrics
    avg_rdia = {}
    for metric, values in rdia_metrics.items():
        if values:
            avg_rdia[metric] = float(np.mean(values))

    # Combine all metrics
    current_results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project_metrics_by_k": {str(k): v for k, v in avg_project_by_k.items()},
        "interest_metrics_by_k": {str(k): v for k, v in avg_interest_by_k.items()},
        "application_metrics_by_k": {str(k): v for k, v in avg_application_by_k.items()},
        "rdia_metrics": avg_rdia,
        "catalog_coverage": coverage,
        "unique_recommended": len(unique_recommended),
        "intersection_with_projects": len(intersection),
        "total_projects_in_folder": len(all_projects_set)
    }

    display_results_with_history(current_results)

# ------------------------------------------------------------
# DISPLAY RESULTS WITH HISTORY
# ------------------------------------------------------------

def display_results_with_history(current_results):

    history = load_history()
    run_number = len(history) + 1

    print("\n" + "="*70)
    print(f" RUN #{run_number} - {current_results['timestamp']}")
    print("="*70)

    if history:
        previous_run = history[-1]
        print("\n COMPARED TO PREVIOUS RUN (Run #{})".format(len(history)))
        print("-" * 70)

        improvements = 0
        regressions = 0

        # Compare projects
        for k in K_VALUES_PROJECTS:
            k_str = str(k)
            print(f"\n  Projects K={k}:")
            current_k = current_results["project_metrics_by_k"][k_str]
            prev_k = previous_run["project_metrics_by_k"][k_str]

            for metric in ["precision", "recall", "ndcg", "mrr"]:
                if metric in current_k and metric in prev_k:
                    current_val = current_k[metric]
                    prev_val = prev_k[metric]
                    diff = current_val - prev_val

                    if abs(diff) < 0.001:
                        status = "o"
                    elif diff > 0:
                        status = "+"
                        improvements += 1
                    else:
                        status = "-"
                        regressions += 1

                    print(f"    {metric:<10}: {status} {diff:+.4f} ({current_val:.4f} vs {prev_val:.4f})")

        # Compare interests
        print(f"\n Interest Metrics:")
        for k in K_VALUES_INTEREST_APP:
            k_str = str(k)
            current_k = current_results["interest_metrics_by_k"][k_str]
            prev_k = previous_run["interest_metrics_by_k"][k_str]

            for metric in ["precision", "recall", "ndcg", "mrr"]:
                if metric in current_k and metric in prev_k:
                    current_val = current_k[metric]
                    prev_val = prev_k[metric]
                    diff = current_val - prev_val

                    if abs(diff) < 0.001:
                        status = "o"
                    elif diff > 0:
                        status = "+"
                        improvements += 1
                    else:
                        status = "-"
                        regressions += 1

                    print(f"    {metric}@{k:<2}: {status} {diff:+.4f} ({current_val:.4f} vs {prev_val:.4f})")

        # Compare applications
        print(f"\n   Application Metrics:")
        for k in K_VALUES_INTEREST_APP:
            k_str = str(k)
            current_k = current_results["application_metrics_by_k"][k_str]
            prev_k = previous_run["application_metrics_by_k"][k_str]

            for metric in ["precision", "recall", "ndcg", "mrr"]:
                if metric in current_k and metric in prev_k:
                    current_val = current_k[metric]
                    prev_val = prev_k[metric]
                    diff = current_val - prev_val

                    if abs(diff) < 0.001:
                        status = "o"
                    elif diff > 0:
                        status = "+"
                        improvements += 1
                    else:
                        status = "-"
                        regressions += 1

                    print(f"    {metric}@{k:<2}: {status} {diff:+.4f} ({current_val:.4f} vs {prev_val:.4f})")

        # Compare RDIA
        print(f"\n   RDIA Metrics:")
        for metric in ["hit_rate", "rank", "mrr"]:
            if metric in current_results["rdia_metrics"] and metric in previous_run["rdia_metrics"]:
                current_val = current_results["rdia_metrics"][metric]
                prev_val = previous_run["rdia_metrics"][metric]
                diff = current_val - prev_val

                if abs(diff) < 0.001:
                    status = "o"
                elif metric == "rank":
                    if diff < 0:
                        status = "+"
                        improvements += 1
                    else:
                        status = "-"
                        regressions += 1
                else:
                    if diff > 0:
                        status = "+"
                        improvements += 1
                    else:
                        status = "-"
                        regressions += 1

                print(f"    {metric:<10}: {status} {diff:+.4f} ({current_val:.4f} vs {prev_val:.4f})")

        # Compare coverage
        if "catalog_coverage" in previous_run:
            current_cov = current_results["catalog_coverage"]
            prev_cov = previous_run["catalog_coverage"]
            diff = current_cov - prev_cov

            if abs(diff) < 0.001:
                status = "o"
            elif diff > 0:
                status = "+"
                improvements += 1
            else:
                status = "-"
                regressions += 1

            print(f"\n Catalog Coverage: {status} {diff:+.4f} ({current_cov:.4f} vs {prev_cov:.4f})")

        print(f"\n Improvements: {improvements}  |  Regressions: {regressions}  |  No Change: {len(['precision', 'recall', 'ndcg', 'mrr']) - improvements - regressions}")

    # Display current results
    print("\n" + "="*70)
    print(" CURRENT RESULTS")
    print("="*70)

    # Project metrics
    print("\n PROJECT METRICS:")
    print("-" * 70)

    header = "Metric".ljust(15)
    for k in K_VALUES_PROJECTS:
        header += f" | K={k}".ljust(12)
    print(header)
    print("-" * 70)

    metrics_order = ["precision", "recall", "ndcg", "mrr"]
    for metric in metrics_order:
        row = metric.ljust(15)
        for k in K_VALUES_PROJECTS:
            k_str = str(k)
            if metric in current_results["project_metrics_by_k"][k_str]:
                val = current_results["project_metrics_by_k"][k_str][metric]
                row += f" | {val:.4f}".ljust(12)
            else:
                row += " | -".ljust(12)
        print(row)

    # Catalog coverage
    print(f"\n CATALOG COVERAGE (based on {current_results['total_projects_in_folder']} projects):")
    print("-" * 50)
    print(f"coverage                 : {current_results['catalog_coverage']:.4f}")
    print(f"unique recommended       : {current_results['unique_recommended']}")
    print(f"intersected with projects: {current_results['intersection_with_projects']}")

    # Interest metrics
    print("\n INTEREST METRICS:")
    print("-" * 70)

    header = "Metric".ljust(15)
    for k in K_VALUES_INTEREST_APP:
        header += f" | K={k}".ljust(12)
    print(header)
    print("-" * 70)

    metrics_order = ["precision", "recall", "ndcg", "mrr"]
    for metric in metrics_order:
        row = metric.ljust(15)
        for k in K_VALUES_INTEREST_APP:
            k_str = str(k)
            if metric in current_results["interest_metrics_by_k"][k_str]:
                val = current_results["interest_metrics_by_k"][k_str][metric]
                row += f" | {val:.4f}".ljust(12)
            else:
                row += " | -".ljust(12)
        print(row)

    # Application metrics
    print("\n APPLICATION METRICS:")
    print("-" * 70)

    header = "Metric".ljust(15)
    for k in K_VALUES_INTEREST_APP:
        header += f" | K={k}".ljust(12)
    print(header)
    print("-" * 70)

    metrics_order = ["precision", "recall", "ndcg", "mrr"]
    for metric in metrics_order:
        row = metric.ljust(15)
        for k in K_VALUES_INTEREST_APP:
            k_str = str(k)
            if metric in current_results["application_metrics_by_k"][k_str]:
                val = current_results["application_metrics_by_k"][k_str][metric]
                row += f" | {val:.4f}".ljust(12)
            else:
                row += " | -".ljust(12)
        print(row)

    # RDIA metrics
    print("\n RDIA METRICS:")
    print("-" * 40)
    for metric, value in current_results["rdia_metrics"].items():
        if metric == "rank":
            print(f"{metric:<15}: {value:.2f} (average rank)")
        else:
            print(f"{metric:<15}: {value:.4f}")

    save_to_history(current_results, history)
    update_baseline(current_results)

# ------------------------------------------------------------
# HISTORY MANAGEMENT
# ------------------------------------------------------------

def load_history():
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_to_history(current_results, history):
    history.append(current_results)
    if len(history) > 20:
        history = history[-20:]

    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\n Results saved to history (Run #{len(history)})")

def update_baseline(current_results):
    baseline_data = {
        "last_run": current_results["timestamp"],
        "run_number": len(load_history()),
        "project_metrics_by_k": current_results["project_metrics_by_k"],
        "interest_metrics_by_k": current_results["interest_metrics_by_k"],
        "application_metrics_by_k": current_results["application_metrics_by_k"],
        "rdia_metrics": current_results["rdia_metrics"],
        "catalog_coverage": current_results["catalog_coverage"],
        "unique_recommended": current_results["unique_recommended"],
        "intersection_with_projects": current_results["intersection_with_projects"],
        "total_projects_in_folder": current_results["total_projects_in_folder"]
    }

    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)

    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    evaluate()
