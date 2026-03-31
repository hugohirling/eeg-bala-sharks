"""
Statistical Testing Module

This module applies parametric and non-parametric statistical methods to evaluate 
behavioral heuristics and electrophysiological (EEG) decoding accuracies. 
It features standard general linear models (T-tests, ANOVA) for scalar metrics, 
as well as Non-Parametric Cluster-Based Permutation Tests, which are the 
"gold standard" in EEG research for mitigating the multiple-comparisons 
problem across continuous time-series data.
"""

import argparse
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# Add the root directory to sys.path to import paths.py configuration
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import paths


# =============================================================================
# EFFECT SIZE CALCULATIONS
# =============================================================================

def cohens_d(group1, group2):
    """
    Calculate Cohen's d to quantify the effect size between two independent groups.
    
    While p-values indicate if an effect exists, Cohen's d indicates how *large* 
    the effect actually is (0.2 = small, 0.5 = medium, 0.8 = large).
    
    Parameters
    ----------
    group1, group2 : array-like
        The two arrays of numerical data to compare.
        
    Returns
    -------
    d : float
        The standardized mean difference.
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Calculate pooled standard deviation (weighted average of standard deviations)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Standardized difference between means
    return (np.mean(group1) - np.mean(group2)) / pooled_sd


# =============================================================================
# PARAMETRIC TESTS (GENERAL LINEAR MODELS)
# =============================================================================

def ttest_paired(cond1, cond2, alpha=0.05):
    """
    Execute a paired (dependent) samples t-test.
    Useful for within-subject designs (e.g., matching Player 1's baseline vs Task).
    
    Returns
    -------
    results : dict
        Contains the t-statistic, p-value, Cohen's d, and boolean significance mask.
    """
    t_stat, p_value = stats.ttest_rel(cond1, cond2)
    
    # For paired data, effect size is calculated on the differences
    diff = np.array(cond1) - np.array(cond2)
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
    
    return {'t': t_stat, 'p': p_value, 'd': d, 'significant': p_value < alpha}


def ttest_independent(group1, group2, alpha=0.05):
    """
    Execute an independent samples t-test.
    Useful for between-group designs (e.g., comparing Player 1 scores vs Player 2 scores).
    """
    t_stat, p_value = stats.ttest_ind(group1, group2)
    d = cohens_d(group1, group2)
    
    return {'t': t_stat, 'p': p_value, 'd': d, 'significant': p_value < alpha}


def anova_oneway(*groups, alpha=0.05):
    """
    Perform a One-Way Analysis of Variance (ANOVA).
    Evaluates variances across three or more independent groups to see if at least 
    one group statistically differs from the others.
    
    Returns
    -------
    results : dict
        Contains F-statistic, p-value, and Eta-squared (variance explanation ratio).
    """
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Calculate Eta-squared (η²) for effect size
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = np.sum((all_data - grand_mean)**2)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    return {'F': f_stat, 'p': p_value, 'eta_squared': eta_squared, 'significant': p_value < alpha}


# =============================================================================
# NON-PARAMETRIC CLUSTER PERMUTATION TESTS (FOR CONTINUOUS EEG)
# =============================================================================

def cluster_permutation_1samp(data, n_permutations=1000, alpha=0.05):
    """
    One-sample cluster-based permutation test.
    
    Given the thousands of timepoints evaluated in M/EEG decoding, conducting 
    individual t-tests creates massive Type 1 error inflation (Family-Wise Error Rate). 
    This test randomly shuffles data labels 1,000 times to create a null distribution 
    acting as a mathematical threshold for true, sustained temporal clusters.
    
    Parameters
    ----------
    data : np.ndarray
        Differences from chance level (e.g., Decoding Accuracy - 0.33) across time.
        Shape: (n_subjects, n_timepoints).
        
    Returns
    -------
    results : dict
        Contains statistical cluster coordinates and their corrected p-values.
    """
    from mne.stats import permutation_cluster_1samp_test
    
    # tail=0 implies a two-tailed test
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
        data, n_permutations=n_permutations, tail=0, verbose=False
    )
    
    # Isolate only the clusters that survived permutation thresholding
    sig_clusters = [(i, cluster_pv[i]) for i in range(len(clusters)) if cluster_pv[i] < alpha]
    
    return {
        't_observed': t_obs,
        'clusters': clusters,
        'cluster_p_values': cluster_pv,
        'significant_clusters': sig_clusters,
        'n_significant': len(sig_clusters)
    }


def cluster_permutation_2samp(cond1, cond2, n_permutations=1000, alpha=0.05):
    """
    Two-sample (Independent) cluster-based permutation test.
    Used to locate temporal windows where two distinct continuous arrays 
    (e.g., Player 1 brain waves vs Player 2 brain waves) significantly diverge.
    """
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
    Ingest scalar behavioral data from all processed subjects and execute 
    group-level (n_subjects > 1) parametric tests. Allows researchers to 
    validate broad cognitive theories across the whole participant cohort.
    """
    print("\nStarting Group-Level Statistical Testing...")
    
    beh_dir = paths.OUTPUT_DIR / "analysis" / "behavioral"
    stats_dir = paths.OUTPUT_DIR / "analysis" / "statistical"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Aggregate Behavioral Data Outputted from script 07
    data_list = []
    if beh_dir.exists():
        for file in beh_dir.glob("*_metrics.json"):
            with open(file, 'r') as f:
                metrics = json.load(f)
                
                # Parse filename to extract BIDS identifiers: e.g., sub-01_P1_metrics.json
                parts = file.stem.split('_')
                metrics['subject'] = parts[0]
                metrics['player'] = parts[1]
                data_list.append(metrics)
                
    df = pd.DataFrame(data_list)
    
    if df.empty:
        print("ERROR: No behavioral metrics found. Run 07_behavioral_analysis.py first.")
        return
        
    print(f"Loaded behavioral data for {len(df)} subject-player sessions.")
    
    # Export grouped spreadsheet for potential external statistical software (e.g., SPSS/JASP)
    group_csv = stats_dir / "group_behavioral_metrics.csv"
    df.to_csv(group_csv, index=False)
    print(f"Aggregated data saved to: {group_csv.name}")
    
    # 2. Example Statistical Run: Did Player 1 or Player 2 utilize the 
    # Win-Stay reinforcement heuristic more often?
    p1_data = df[df['player'] == 'P1']['win_stay_rate'].dropna()
    p2_data = df[df['player'] == 'P2']['win_stay_rate'].dropna()
    
    if len(p1_data) > 0 and len(p2_data) > 0:
        t_stat, p_val = stats.ttest_ind(p1_data, p2_data)
        eff_size = cohens_d(p1_data, p2_data)
        
        print("\n--- T-Test Result: Win-Stay Rate (P1 vs P2) ---")
        print(f"T-statistic: {t_stat:.3f}")
        print(f"P-value:     {p_val:.4f}")
        print(f"Cohen's d:   {eff_size:.3f}")
        print("-----------------------------------------------")
        
    print("\nFinished Statistical Testing.")

if __name__ == "__main__":
    run_group_statistics()