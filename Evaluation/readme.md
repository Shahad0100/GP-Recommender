# EvaluationMatrices.py - Simple Guide

## 📋 What This File Does
This file checks if your recommendation system is working well by comparing what it suggests vs what it should suggest.

## 🔍 The Measurements (Metrics)

### For Projects (K=3,5,7,10)
| Metric | What it means | Why we use it |
|--------|---------------|---------------|
| **Precision@K** | Out of 10 suggestions, how many were correct? | Measures accuracy |
| **Recall@K** | Out of all correct items, how many did we find? | Measures coverage |
| **MAP@K** | Average correctness across all positions | Overall quality |
| **NDCG@K** | Correct items at the top matter more | Good ranking is important |
| **α-NDCG@K** | Balances being correct and being different | Prevents same-old suggestions |
| **ILD** | How different are the suggestions? | Ensures variety |
| **MRR** | Where does the first correct item appear? | How fast we get it right |

### For Interests & Applications (K=1,3)
Same idea but for interests and app categories

### For RDIA (Research Type)
- **Accuracy**: Did we guess the research type correctly?
- **F1 Score**: Balance between being correct and finding all
- **MRR**: Rank of first correct guess

### Catalog Coverage
- How many different projects did we suggest overall?
- Uses all 43 available projects as the total pool

## 🔄 How We Compare

1. **First time you run it**: Saves results as baseline
2. **Next times**: Shows what changed since last run
3. **Difference** = New value - Old value
   - Higher number = Better
   - Lower number = Worse
   - Same number = No change

## 📁 Files Created

- `evaluation_history.json`: Last 20 runs (to see progress)
- `evaluation_baseline.json`: Most recent run (current status)

## 🚀 How to Run It

1. **Make sure these exist**:
   - Ground truth file (correct answers)
   - Your model's results file
   - Folder with all 43 projects

2. **Run the script**:
   ```
   python EvaluationMatrices.py
   ```

3. **Look at the output**:
   - See current scores
   - Check if better or worse than last time
   - Find weak spots to improve

## 🎯 Why These Measurements?

Each one tells us something different:
- **Precision/Recall**: Basic - did we get it right?
- **NDCG**: Is the best stuff at the top?
- **α-NDCG**: Are we suggesting new things or the same old?
- **ILD**: Are suggestions different from each other?
- **Coverage**: Are we using all projects or just a few?