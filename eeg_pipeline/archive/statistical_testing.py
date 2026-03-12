"""
Statistical Testing Module
- T-tests and ANOVA
- Cluster-based permutation tests

Author: Ayush
"""

import numpy as np
from scipy import stats


# =============================================================================
# EFFECT SIZE
# =============================================================================

def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size.
    
    Returns
    -------
    d : float
        Cohen's d (0.2=small, 0.5=medium, 0.8=large)
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


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