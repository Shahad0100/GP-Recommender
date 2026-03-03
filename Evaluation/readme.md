# EvaluationMatrices.py - Simple Guide

## What This File Does
This script evaluates the performance of your project recommendation system. It compares the system's suggestions (from `all_results.json`) against the established "ground truth" (from `Experimental Groups with Recommendations.json`) using a variety of ranking and classification metrics.

## The Measurements (Metrics)

### For Projects (evaluated at K=3, 5, 7, 10)
These metrics measure how well the system recommends projects.

| Metric | What it means | Why we use it |
|--------|---------------|---------------|
| **Precision@K** | Of the top-K projects recommended, how many are relevant? | Measures the accuracy of the recommendations. |
| **Recall@K** | Of all the relevant projects for a group, how many appear in the top-K? | Measures the system's ability to find all relevant items. |
| **MAP@K** (Mean Average Precision) | The average precision across all relevant projects, considering their rank. | Provides a single-figure measure of quality across all ranks. |
| **NDCG@K** (Normalized Discounted Cumulative Gain) | Rewards relevant projects that appear higher in the list. It assumes that users are more likely to see and prefer top-ranked items. | Evaluates the quality of the ranking order. |
| **α-NDCG@K** (Alpha-NDCG) | An extension of NDCG that also rewards diversity. It reduces the gain for a recommended project if its topics (interests) have already been covered by higher-ranked projects. | Prevents the system from suggesting projects that are all very similar to each other. |
| **ILD** (Intra-List Diversity) | Measures how different the recommended projects are from one another, based on their associated interests. A higher score means more diverse recommendations. | Ensures the recommendation list provides variety. |
| **MRR** (Mean Reciprocal Rank) | Looks at the rank position of the *first* relevant project in the list. It answers the question: "How soon does the system get it right?" | Good for scenarios where the user only cares about the first good result. |

### For Interests & Applications (evaluated at K=1, 3)
These metrics measure how well the system predicts the group's preferred topics and application domains.

- **Precision@K, Recall@K, NDCG@K, MRR**: These metrics work the same way as they do for projects, but are applied to the lists of recommended `interests` and `applications`.

### For RDIA (Research, Development, and Innovation Authority categories)
This measures how well the system predicts the single, most relevant RDIA category for a group. The ground truth has **one** value, while the system recommends **four** ranked categories. The evaluation is now tailored to this scenario:

- **Hit Rate**: Does the single correct RDIA category appear anywhere in the list of 4 recommendations? (Expressed as a value between 0 and 1, or a percentage).
- **Average Rank**: If the correct category was found, what was its average position (1 being the best)? If it was not found, it's counted as being ranked 5th (outside the list). **For this metric, a lower number is better.**
- **MRR (Mean Reciprocal Rank)**: The average of `1/rank` for the correct category. This rewards finding the correct answer early (e.g., 1/1 = 1.0) and penalizes not finding it (score of 0).

### Catalog Coverage
- **What it is**: The percentage of unique projects from the entire project catalog (all 43 projects in the `data/projects` folder) that were recommended to *any* group in the top-K (using the maximum K=10).
- **Why we use it**: To see if the system is exploring the full range of available projects or just recommending a small, popular subset.

## How We Compare

1.  **First time you run it**: The script creates a baseline by saving the results to `evaluation_history.json`.
2.  **Subsequent runs**: The script loads the previous run's results and shows what has changed.
3.  **Change (Δ)**: Calculated as `New Value - Old Value`.
    - **For most metrics (Precision, Recall, NDCG, Hit Rate, MRR, Coverage)**: A **higher (positive) change** is better.
    - **For the `rank` metric (RDIA)**: A **lower (negative) change** is better, as it means the correct category is appearing earlier in the list.

## Files Created

- `evaluation_history.json`: Stores the results of the last 20 runs to track progress over time.
- `evaluation_baseline.json`: Stores the results of the most recent run, serving as the current benchmark.

## How to Run It

1.  **Prerequisites**: Ensure the following files and folders exist with the correct data:
    - `Silver Ground Truth/Experimental Groups with Recommendations.json`
    - `test_results/all_results.json`
    - `data/projects/` (folder containing the 43 project JSON files)
2.  **Run the script**:
    ```bash
    python EvaluationMatrices.py
