import pandas as pd
from sklearn.metrics import cohen_kappa_score
import os

def run_kappa_analysis():
    print("--- Cohen's Kappa Analysis ---")
    if not os.path.exists("phase-b/human_labels.csv"):
        print("Error: human_labels.csv not found.")
        return
    
    if not os.path.exists("phase-b/pairwise_results.csv"):
        print("Error: pairwise_results.csv not found.")
        return

    human_df = pd.read_csv("phase-b/human_labels.csv")
    judge_df = pd.read_csv("phase-b/pairwise_results.csv")

    # Align data
    human_labels = human_df['human_winner'].tolist()
    judge_labels = judge_df['winner_after_swap'].head(len(human_labels)).tolist()

    kappa = cohen_kappa_score(human_labels, judge_labels)
    print(f"Cohen's Kappa: {kappa:.3f}")

    # Interpretation
    if kappa < 0.2:
        interpretation = "Slight agreement"
    elif kappa < 0.4:
        interpretation = "Fair agreement"
    elif kappa < 0.6:
        interpretation = "Moderate agreement"
    elif kappa < 0.8:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"
    
    print(f"Interpretation: {interpretation}")

if __name__ == "__main__":
    run_kappa_analysis()
