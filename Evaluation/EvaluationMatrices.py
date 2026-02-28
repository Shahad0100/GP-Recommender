import os
import json
import numpy as np
from collections import defaultdict
from datetime import datetime
import glob

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

SILVER_GT_PATH = r"C:\Users\dhekr\Desktop\GP-Recommender\Silver Ground Truth\Experimental Groups with Recommendations.json"
MODEL_RESULTS_PATH = r"C:\Users\dhekr\Desktop\GP-Recommender\test_results\all_results.json"
PROJECTS_FOLDER_PATH = r"C:\Users\dhekr\Desktop\GP-Recommender\data\projects"  # مجلد المشاريع
BASELINE_PATH = r"C:\Users\dhekr\Desktop\GP-Recommender\Evaluation\evaluation_baseline.json"
HISTORY_PATH = r"C:\Users\dhekr\Desktop\GP-Recommender\Evaluation\evaluation_history.json"

# قيم K المختلفة للتقييم
K_VALUES_PROJECTS = [3, 5, 7, 10]  # للمشاريع
K_VALUES_INTEREST_APP = [1, 3]      # للاهتمامات والتطبيقات

ALPHA = 0.5   # for α-NDCG

# ─────────────────────────────────────────────
# BASIC METRICS
# ─────────────────────────────────────────────

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

def average_precision(rec, rel, k):
    score = 0
    hits = 0
    for i, r in enumerate(rec[:k], 1):
        if r in rel:
            hits += 1
            score += hits / i
    return score / min(len(rel), k) if rel else 0

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
    """Accuracy: correct predictions / total predictions"""
    if not rec or not rel:
        return 0
    correct = len(set(rec) & set(rel))
    return correct / len(rec) if rec else 0

def f1_score(rec, rel):
    """F1 score = 2 * (precision * recall) / (precision + recall)"""
    if not rec or not rel:
        return 0
    rec_set = set(rec)
    rel_set = set(rel)
    
    precision = len(rec_set & rel_set) / len(rec_set) if rec_set else 0
    recall = len(rec_set & rel_set) / len(rel_set) if rel_set else 0
    
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

# ─────────────────────────────────────────────
# α-NDCG
# ─────────────────────────────────────────────

def alpha_ndcg(rec_projects, gt_interests, k, alpha=0.5):
    covered = defaultdict(int)
    score = 0
    
    for i, project in enumerate(rec_projects[:k], 1):
        gain = 0
        project_interests = project.get("interest", [])
        
        for interest in project_interests:
            if interest in gt_interests:
                gain += (1 - alpha) ** covered[interest]
                covered[interest] += 1
                
        score += gain / np.log2(i + 1)
    
    ideal = len(gt_interests)
    return score / ideal if ideal > 0 else 0

# ─────────────────────────────────────────────
# ILD (Intra-List Diversity)
# ─────────────────────────────────────────────

def intra_list_diversity(projects, k):
    """Calculate diversity based on interest categories"""
    if not projects or len(projects) < 2:
        return 0
    
    interest_sets = []
    for p in projects[:k]:
        interest_set = set(p.get("interest", []))
        if interest_set:
            interest_sets.append(interest_set)
    
    if len(interest_sets) < 2:
        return 0
    
    similarities = []
    for i in range(len(interest_sets)):
        for j in range(i + 1, len(interest_sets)):
            if interest_sets[i] and interest_sets[j]:
                jaccard = len(interest_sets[i] & interest_sets[j]) / len(interest_sets[i] | interest_sets[j])
                similarities.append(jaccard)
            else:
                similarities.append(0)
    
    return 1 - np.mean(similarities) if similarities else 0

# ─────────────────────────────────────────────
# Catalog Coverage - المعدل حسب الـ 43 ملف
# ─────────────────────────────────────────────

def get_all_projects_from_folder():
    """
    قراءة جميع أسماء ملفات المشاريع من المجلد
    """
    project_files = glob.glob(os.path.join(PROJECTS_FOLDER_PATH, "*.json"))
    project_ids = []
    
    for file_path in project_files:
        # استخراج اسم الملف بدون امتداد .json
        file_name = os.path.basename(file_path)
        project_id = file_name.replace('.json', '')
        project_ids.append(project_id)
    
    print(f"📁 Found {len(project_ids)} project files in {PROJECTS_FOLDER_PATH}")
    return set(project_ids)

def catalog_coverage(recommended_projects, all_projects_set):
    """
    Percentage of unique projects recommended across all groups
    مقارنة مع جميع المشاريع الموجودة في مجلد data/projects
    """
    unique_recommended = set()
    
    for group_recs in recommended_projects:
        for rec in group_recs[:K_VALUES_PROJECTS[-1]]:  # Use max K
            unique_recommended.add(rec["project_id"])
    
    # حساب التقاطع بين المشاريع الموصى بها وجميع المشاريع الموجودة
    intersection = unique_recommended & all_projects_set
    coverage = len(intersection) / len(all_projects_set) if all_projects_set else 0
    
    return coverage, unique_recommended, intersection

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_silver_gt():
    with open(SILVER_GT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {g["group_id"]: g for g in data}

def load_model_results():
    with open(MODEL_RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {g["group_id"]: g for g in data["results"]}

# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────

def evaluate():
    print("Loading data...")
    silver = load_silver_gt()
    results = load_model_results()
    
    print(f"Loaded {len(silver)} groups from silver GT")
    print(f"Loaded {len(results)} groups from model results")
    
    # الحصول على جميع المشاريع من مجلد data/projects
    all_projects_set = get_all_projects_from_folder()
    print(f"Total projects in folder: {len(all_projects_set)}")
    
    # تخزين المقاييس لكل قيمة K
    project_metrics_by_k = {k: defaultdict(list) for k in K_VALUES_PROJECTS}
    interest_metrics_by_k = {k: defaultdict(list) for k in K_VALUES_INTEREST_APP}
    application_metrics_by_k = {k: defaultdict(list) for k in K_VALUES_INTEREST_APP}
    rdia_metrics = defaultdict(list)
    
    all_recommendations = []
    
    for gid, gt in silver.items():
        if gid not in results:
            print(f"Warning: group {gid} not in model results, skipping.")
            continue
        
        # Ground truth data
        gt_projects = [p["project_id"] for p in gt.get("recommended_projects", [])]
        group_profile = results[gid].get("group_profile", {})
        gt_interests = group_profile.get("selected_interests", [])
        gt_applications = group_profile.get("selected_applications", [])
        gt_rdia = group_profile.get("selected_rdia", [])
        
        # Model recommendations
        rec_projects = results[gid].get("recommended_projects", [])
        rec_ids = [p["project_id"] for p in rec_projects]
        all_recommendations.append(rec_projects)
        
        # Recommended interests/applications/rdia
        rec_interests = [i["name"] for i in results[gid].get("recommended_interests", [])[:K_VALUES_INTEREST_APP[-1]]]
        rec_applications = [a["name"] for a in results[gid].get("recommended_applications", [])[:K_VALUES_INTEREST_APP[-1]]]
        rec_rdia = [r["label"] for r in results[gid].get("recommended_rdia", [])[:len(gt_rdia)]]
        
        # ─── PROJECT METRICS FOR EACH K ──────────
        for k in K_VALUES_PROJECTS:
            project_metrics_by_k[k]["precision"].append(precision_at_k(rec_ids, gt_projects, k))
            project_metrics_by_k[k]["recall"].append(recall_at_k(rec_ids, gt_projects, k))
            project_metrics_by_k[k]["map"].append(average_precision(rec_ids, gt_projects, k))
            project_metrics_by_k[k]["ndcg"].append(ndcg_at_k(rec_ids, gt_projects, k))
            project_metrics_by_k[k]["alpha_ndcg"].append(alpha_ndcg(rec_projects, gt_interests, k, ALPHA))
            project_metrics_by_k[k]["ild"].append(intra_list_diversity(rec_projects, k))
        
        # MRR (independent of K)
        project_metrics_by_k[K_VALUES_PROJECTS[0]]["mrr"].append(mrr(rec_ids, gt_projects))
        
        # ─── INTEREST METRICS FOR EACH K ─────────
        if gt_interests:
            for k in K_VALUES_INTEREST_APP:
                interest_metrics_by_k[k]["precision"].append(precision_at_k(rec_interests, gt_interests, k))
                interest_metrics_by_k[k]["recall"].append(recall_at_k(rec_interests, gt_interests, k))
                interest_metrics_by_k[k]["ndcg"].append(ndcg_at_k(rec_interests, gt_interests, k))
        
        # MRR for interests (independent of K)
        interest_metrics_by_k[K_VALUES_INTEREST_APP[0]]["mrr"].append(mrr(rec_interests, gt_interests))
        
        # ─── APPLICATION METRICS FOR EACH K ───────
        if gt_applications:
            for k in K_VALUES_INTEREST_APP:
                application_metrics_by_k[k]["precision"].append(precision_at_k(rec_applications, gt_applications, k))
                application_metrics_by_k[k]["recall"].append(recall_at_k(rec_applications, gt_applications, k))
                application_metrics_by_k[k]["ndcg"].append(ndcg_at_k(rec_applications, gt_applications, k))
        
        # MRR for applications (independent of K)
        application_metrics_by_k[K_VALUES_INTEREST_APP[0]]["mrr"].append(mrr(rec_applications, gt_applications))
        
        # ─── RDIA METRICS ────────────────────────
        if gt_rdia:
            rdia_metrics["accuracy"].append(accuracy(rec_rdia, gt_rdia))
            rdia_metrics["f1"].append(f1_score(rec_rdia, gt_rdia))
            rdia_metrics["mrr"].append(mrr(rec_rdia, gt_rdia))
    
    # ─────────────────────────────────────────────
    # CALCULATE AVERAGES
    # ─────────────────────────────────────────────
    
    # Project metrics for each K
    avg_project_by_k = {}
    for k in K_VALUES_PROJECTS:
        avg_project_by_k[k] = {}
        for metric, values in project_metrics_by_k[k].items():
            if values:
                avg_project_by_k[k][metric] = float(np.mean(values))
    
    # Add MRR (same for all K)
    mrr_value = float(np.mean(project_metrics_by_k[K_VALUES_PROJECTS[0]]["mrr"])) if project_metrics_by_k[K_VALUES_PROJECTS[0]]["mrr"] else 0
    for k in K_VALUES_PROJECTS:
        avg_project_by_k[k]["mrr"] = mrr_value
    
    # Interest metrics for each K
    avg_interest_by_k = {}
    for k in K_VALUES_INTEREST_APP:
        avg_interest_by_k[k] = {}
        for metric, values in interest_metrics_by_k[k].items():
            if values:
                avg_interest_by_k[k][metric] = float(np.mean(values))
    
    # Add MRR for interests (same for all K)
    mrr_interest = float(np.mean(interest_metrics_by_k[K_VALUES_INTEREST_APP[0]]["mrr"])) if interest_metrics_by_k[K_VALUES_INTEREST_APP[0]]["mrr"] else 0
    for k in K_VALUES_INTEREST_APP:
        avg_interest_by_k[k]["mrr"] = mrr_interest
    
    # Application metrics for each K
    avg_application_by_k = {}
    for k in K_VALUES_INTEREST_APP:
        avg_application_by_k[k] = {}
        for metric, values in application_metrics_by_k[k].items():
            if values:
                avg_application_by_k[k][metric] = float(np.mean(values))
    
    # Add MRR for applications (same for all K)
    mrr_application = float(np.mean(application_metrics_by_k[K_VALUES_INTEREST_APP[0]]["mrr"])) if application_metrics_by_k[K_VALUES_INTEREST_APP[0]]["mrr"] else 0
    for k in K_VALUES_INTEREST_APP:
        avg_application_by_k[k]["mrr"] = mrr_application
    
    # Catalog coverage - باستخدام الـ 43 ملف
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
    
    # عرض النتائج
    display_results_with_history(current_results)

# ─────────────────────────────────────────────
# DISPLAY RESULTS WITH HISTORY
# ─────────────────────────────────────────────

def display_results_with_history(current_results):
    """عرض النتائج مع السجل التاريخي"""
    
    history = load_history()
    run_number = len(history) + 1
    
    print("\n" + "="*70)
    print(f"🏃‍♂️ RUN #{run_number} - {current_results['timestamp']}")
    print("="*70)
    
    # عرض المقارنة مع التشغلة السابقة
    if history:
        previous_run = history[-1]
        print("\n📊 COMPARED TO PREVIOUS RUN (Run #{})".format(len(history)))
        print("-" * 70)
        
        # مقارنة لمختلف قيم K للمشاريع
        improvements = 0
        regressions = 0
        
        # مقارنة المشاريع
        for k in K_VALUES_PROJECTS:
            k_str = str(k)
            print(f"\n  📁 Projects K={k}:")
            current_k = current_results["project_metrics_by_k"][k_str]
            prev_k = previous_run["project_metrics_by_k"][k_str]
            
            for metric in ["precision", "recall", "map", "ndcg", "alpha_ndcg", "ild", "mrr"]:
                if metric in current_k and metric in prev_k:
                    current_val = current_k[metric]
                    prev_val = prev_k[metric]
                    diff = current_val - prev_val
                    
                    if abs(diff) < 0.001:
                        status = "⚪"
                    elif diff > 0:
                        status = "🟢"
                        improvements += 1
                    else:
                        status = "🔴"
                        regressions += 1
                    
                    print(f"    {metric:<10}: {status} {diff:+.4f} ({current_val:.4f} vs {prev_val:.4f})")
        
        # مقارنة الاهتمامات
        print(f"\n  🎯 Interest Metrics:")
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
                        status = "⚪"
                    elif diff > 0:
                        status = "🟢"
                        improvements += 1
                    else:
                        status = "🔴"
                        regressions += 1
                    
                    print(f"    {metric}@{k:<2}: {status} {diff:+.4f} ({current_val:.4f} vs {prev_val:.4f})")
        
        # مقارنة التطبيقات
        print(f"\n  📱 Application Metrics:")
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
                        status = "⚪"
                    elif diff > 0:
                        status = "🟢"
                        improvements += 1
                    else:
                        status = "🔴"
                        regressions += 1
                    
                    print(f"    {metric}@{k:<2}: {status} {diff:+.4f} ({current_val:.4f} vs {prev_val:.4f})")
        
        # مقارنة RDIA
        print(f"\n  🏷️ RDIA Metrics:")
        for metric in ["accuracy", "f1", "mrr"]:
            if metric in current_results["rdia_metrics"] and metric in previous_run["rdia_metrics"]:
                current_val = current_results["rdia_metrics"][metric]
                prev_val = previous_run["rdia_metrics"][metric]
                diff = current_val - prev_val
                
                if abs(diff) < 0.001:
                    status = "⚪"
                elif diff > 0:
                    status = "🟢"
                    improvements += 1
                else:
                    status = "🔴"
                    regressions += 1
                
                print(f"    {metric:<10}: {status} {diff:+.4f} ({current_val:.4f} vs {prev_val:.4f})")
        
        # مقارنة التغطية
        if "catalog_coverage" in previous_run:
            current_cov = current_results["catalog_coverage"]
            prev_cov = previous_run["catalog_coverage"]
            diff = current_cov - prev_cov
            
            if abs(diff) < 0.001:
                status = "⚪"
            elif diff > 0:
                status = "🟢"
                improvements += 1
            else:
                status = "🔴"
                regressions += 1
            
            print(f"\n  📚 Catalog Coverage: {status} {diff:+.4f} ({current_cov:.4f} vs {prev_cov:.4f})")
        
        print(f"\n  ✅ Improvements: {improvements}  |  🔴 Regressions: {regressions}")
    
    # عرض النتائج الحالية
    print("\n" + "="*70)
    print("📈 CURRENT RESULTS")
    print("="*70)
    
    # Project metrics for each K
    print("\n📊 PROJECT METRICS:")
    print("-" * 70)
    
    # Header
    header = "Metric".ljust(15)
    for k in K_VALUES_PROJECTS:
        header += f" | K={k}".ljust(12)
    print(header)
    print("-" * 70)
    
    # Rows
    metrics_order = ["precision", "recall", "map", "ndcg", "alpha_ndcg", "ild", "mrr"]
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
    
    # Catalog coverage - مع معلومات تفصيلية
    print(f"\n📚 CATALOG COVERAGE (based on 43 projects):")
    print("-" * 50)
    print(f"coverage                 : {current_results['catalog_coverage']:.4f}")
    print(f"unique recommended       : {current_results['unique_recommended']}")
    print(f"intersected with 43 files: {current_results['intersection_with_projects']}")
    print(f"total projects in folder : {current_results['total_projects_in_folder']}")
    
    
    
    # Interest metrics for each K
    print("\n🎯 INTEREST METRICS:")
    print("-" * 70)
    
    # Header
    header = "Metric".ljust(15)
    for k in K_VALUES_INTEREST_APP:
        header += f" | K={k}".ljust(12)
    print(header)
    print("-" * 70)
    
    # Rows
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
    
    # Application metrics for each K
    print("\n📱 APPLICATION METRICS:")
    print("-" * 70)
    
    # Header
    header = "Metric".ljust(15)
    for k in K_VALUES_INTEREST_APP:
        header += f" | K={k}".ljust(12)
    print(header)
    print("-" * 70)
    
    # Rows
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
    print("\n🏷️ RDIA METRICS:")
    print("-" * 40)
    for metric, value in current_results["rdia_metrics"].items():
        print(f"{metric:<15}: {value:.4f}")
    
    # حفظ النتائج
    save_to_history(current_results, history)
    update_baseline(current_results)



# ─────────────────────────────────────────────
# HISTORY MANAGEMENT
# ─────────────────────────────────────────────

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
    
    # تأكد من وجود المجلد
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to history (Run #{len(history)})")

def update_baseline(current_results):
    """تحديث ملف baseline بآخر نتيجة"""
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
    
    # تأكد من وجود المجلد
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    evaluate()
