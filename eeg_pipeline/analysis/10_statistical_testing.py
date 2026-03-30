"""
Statistical Testing Module
- T-tests and ANOVA
- Cluster-based permutation tests

Author: Ayush
"""

import argparse
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# Add the root directory to sys.path to import paths.py
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import paths


# =============================================================================
# EFFECT SIZE
# =============================================================================

def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size.
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Calculate pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    return (np.mean(group1) - np.mean(group2)) / pooled_sd

# =============================================================================
# T-TESTS
# =============================================================================

def ttest_paired(cond1, cond2, alpha=0.05):
    """Paired t-test between two conditions."""
    t_stat, p_value = stats.ttest_rel(cond1, cond2)
    diff = np.array(cond1) - np.array(cond2)
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
    
    return {'t': t_stat, 'p': p_value, 'd': d, 'significant': p_value < alpha}


def ttest_independent(group1, group2, alpha=0.05):
    """Independent samples t-test."""
    t_stat, p_value = stats.ttest_ind(group1, group2)
    d = cohens_d(group1, group2)
    
    return {'t': t_stat, 'p': p_value, 'd': d, 'significant': p_value < alpha}


# =============================================================================
# ANOVA
# =============================================================================

def anova_oneway(*groups, alpha=0.05):
    """One-way ANOVA across multiple groups."""
    f_stat, p_value = stats.f_oneway(*groups)
    
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = np.sum((all_data - grand_mean)**2)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    return {'F': f_stat, 'p': p_value, 'eta_squared': eta_squared, 'significant': p_value < alpha}


# =============================================================================
# CLUSTER PERMUTATION TESTS
# =============================================================================

def cluster_permutation_1samp(data, n_permutations=1000, alpha=0.05):
    """One-sample cluster permutation test."""
    from mne.stats import permutation_cluster_1samp_test
    
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
        data, n_permutations=n_permutations, tail=0, verbose=False
    )
    
    sig_clusters = [(i, cluster_pv[i]) for i in range(len(clusters)) if cluster_pv[i] < alpha]
    
    return {
        't_observed': t_obs,
        'clusters': clusters,
        'cluster_p_values': cluster_pv,
        'significant_clusters': sig_clusters,
        'n_significant': len(sig_clusters)
    }


def cluster_permutation_2samp(cond1, cond2, n_permutations=1000, alpha=0.05):
    """Two-sample cluster permutation test."""
    from mne.stats import permutation_cluster_test
    
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_test(
        [cond1, cond2], n_permutations=n_permutations, tail=0, verbose=False
    )
    
    sig_clusters = [(i, cluster_pv[i]) for i in range(len(clusters)) if cluster_pv[i] < alpha]
    
    return {
        't_observed': t_obs,
        'clusters': clusters,
        'cluster_p_values': cluster_pv,
        'significant_clusters': sig_clusters,
        'n_significant': len(sig_clusters)
    }

# =============================================================================
# PIPELINE INTEGRATION (GROUP LEVEL)
# =============================================================================

def run_group_statistics():
    """
    Load data from all subjects and run group-level statistical tests.
    """
    print("\nStarting Group-Level Statistical Testing...")
    
    beh_dir = paths.OUTPUT_DIR / "analysis" / "behavioral"
    stats_dir = paths.OUTPUT_DIR / "analysis" / "statistical"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Aggregate Behavioral Data
    data_list = []
    if beh_dir.exists():
        for file in beh_dir.glob("*_metrics.json"):
            with open(file, 'r') as f:
                metrics = json.load(f)
                
                # Parse filename: e.g., sub-01_P1_metrics.json
                parts = file.stem.split('_')
                metrics['subject'] = parts[0]
                metrics['player'] = parts[1]
                data_list.append(metrics)
                
    df = pd.DataFrame(data_list)
    
    if df.empty:
        print("ERROR: No behavioral metrics found. Run 07_behavioral_analysis.py first.")
        return
        
    print(f"Loaded behavioral data for {len(df)} subject-player sessions.")
    
    # Save the aggregated group dataframe
    group_csv = stats_dir / "group_behavioral_metrics.csv"
    df.to_csv(group_csv, index=False)
    print(f"Aggregated data saved to: {group_csv.name}")
    
    # 2. Example Statistical Run: Player 1 vs Player 2 Win-Stay Rate
    p1_data = df[df['player'] == 'P1']['win_stay_rate'].dropna()
    p2_data = df[df['player'] == 'P2']['win_stay_rate'].dropna()
    
    if len(p1_data) > 0 and len(p2_data) > 0:
        t_stat, p_val = stats.ttest_ind(p1_data, p2_data)
        eff_size = cohens_d(p1_data, p2_data)
        
        print("\n--- T-Test Result: Win-Stay Rate (P1 vs P2) ---")
        print(f"T-statistic: {t_stat:.3f}")
        print(f"P-value:     {p_val:.4f}")
        print(f"Cohen's d:   {eff_size:.3f}")
        
    print("\nFinished Statistical Testing.")

if __name__ == "__main__":
    # No subject argument needed since this runs on all files in the output folder
    run_group_statistics()