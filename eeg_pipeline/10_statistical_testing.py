"""
Step 10: Statistical Testing
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import config
import warnings
warnings.filterwarnings('ignore')


def run_statistical_tests():
    """Run statistical tests on analysis results."""
    
    results = []
    
    # Load predictability data
    pred_path = config.OUTPUT_DIRS['predictability'] / "predictability_summary.csv"
    if pred_path.exists():
        print("  Loading predictability data...")
        pred_df = pd.read_csv(pred_path)
        
        # Compare P1 vs P2 predictability
        p1_scores = pred_df[pred_df['person'] == 'P1']['predictability_score'].values
        p2_scores = pred_df[pred_df['person'] == 'P2']['predictability_score'].values
        
        if len(p1_scores) > 0 and len(p2_scores) > 0:
            t_stat, p_val = stats.ttest_ind(p1_scores, p2_scores)
            results.append({
                'test': 'Predictability P1 vs P2',
                'statistic': t_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            })
            print(f"    Predictability t-test: t={t_stat:.3f}, p={p_val:.3f}")
    else:
        print(f"  Warning: Predictability data not found at {pred_path}")
    
    # Load PLV data
    plv_path = config.OUTPUT_DIRS['plv'] / "plv_summary.csv"
    if plv_path.exists():
        print("  Loading PLV data...")
        plv_df = pd.read_csv(plv_path)
        
        # Test if real PLV > shuffled PLV
        real_plv = plv_df['plv_real'].values
        shuffled_plv = plv_df['plv_shuffled'].values
        
        if len(real_plv) > 0:
            t_stat, p_val = stats.ttest_rel(real_plv, shuffled_plv)
            results.append({
                'test': 'PLV Real vs Shuffled',
                'statistic': t_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            })
            print(f"    PLV paired t-test: t={t_stat:.3f}, p={p_val:.3f}")
    else:
        print(f"  Warning: PLV data not found at {plv_path}")
    
    # Load time-frequency data
    tf_path = config.OUTPUT_DIRS['time_frequency'] / "time_frequency_summary.csv"
    if tf_path.exists():
        print("  Loading time-frequency data...")
        tf_df = pd.read_csv(tf_path)
        
        # Compare band power between P1 and P2
        for band in config.FREQ_BANDS.keys():
            band_data = tf_df[tf_df['band'] == band]
            p1_power = band_data[band_data['person'] == 'P1']['power'].values
            p2_power = band_data[band_data['person'] == 'P2']['power'].values
            
            if len(p1_power) > 0 and len(p2_power) > 0:
                t_stat, p_val = stats.ttest_ind(p1_power, p2_power)
                results.append({
                    'test': f'{band.capitalize()} Power P1 vs P2',
                    'statistic': t_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                })
    else:
        print(f"  Warning: Time-frequency data not found at {tf_path}")
    
    return results


if __name__ == "__main__":
    print("="*60)
    print("STEP 10: STATISTICAL TESTING")
    print("="*60)
    
    # Ensure output directory exists
    config.OUTPUT_DIRS['statistics'].mkdir(parents=True, exist_ok=True)
    
    print("\nRunning statistical tests...")
    results = run_statistical_tests()
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        out_path = config.OUTPUT_DIRS['statistics'] / "statistical_results_summary.csv"
        df.to_csv(out_path, index=False)
        print(f"\n✓ Saved results to {out_path}")
        print("\n" + df.to_string())
    else:
        print("\n⚠ No statistical results generated")
    
    print("\n" + "="*60)
    print("STEP 10 COMPLETE")
    print("="*60)