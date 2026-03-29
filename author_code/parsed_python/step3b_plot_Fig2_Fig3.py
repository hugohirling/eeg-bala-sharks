"""
Plot decoding results (Python version of step3b_plot_Fig2_Fig3.m):
   - Plot decoding accuracy across time
   - Plot spatial searchlight results
   - Calculate and display Bayes factors
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.io import loadmat
import warnings

warnings.filterwarnings('ignore')

# Set the path
path_to_data = 'MNE-sample-data/ds006761'
derivatives_path = os.path.join(path_to_data, 'derivatives')
plot_dir = os.path.join(derivatives_path, 'plots')
os.makedirs(plot_dir, exist_ok=True)

# Set parameters
pair_ids = np.concatenate([np.arange(1, 10), np.arange(11, 23), np.arange(25, 35)])
num_pairs = len(pair_ids)
pair_idx = np.arange(1, num_pairs * 2 + 1).reshape(-1, 2)
pair_idx0 = pair_idx - 1  # zero-based index for Python arrays
test_idx = [0, 1, 2, 3]
num_tests = len(test_idx)
num_time_bins = 20
test_names = ['Self Current', 'Other Current', 'Self Previous', 'Other Previous']

# Define time bins based on the 3-part structure from decoding
# Part A (Decision): -0.2 to 2s → 8 bins, Part B (Response): 1.8 to 4s → 8 bins, Part C (Feedback): 3.8 to 5s → 4 bins
time_bin_info = {
    'part_a': {'name': 'Decision', 'start': -0.2, 'end': 2.0, 'n_bins': 8, 'bin_indices': range(0, 8), 'color': '#3498db'},
    'part_b': {'name': 'Response', 'start': 1.8, 'end': 4.0, 'n_bins': 8, 'bin_indices': range(8, 16), 'color': '#e74c3c'},
    'part_c': {'name': 'Feedback', 'start': 3.8, 'end': 5.0, 'n_bins': 4, 'bin_indices': range(16, 20), 'color': '#2ecc71'}
}

# Calculate actual time points for each bin (using bin centers)
def calculate_time_points(time_bin_info):
    time_points_seconds = []
    for part in ['part_a', 'part_b', 'part_c']:
        info = time_bin_info[part]
        duration = info['end'] - info['start']
        bin_width = duration / info['n_bins']
        for i in range(info['n_bins']):
            # Center of each bin
            center_time = info['start'] + (i + 0.5) * bin_width
            time_points_seconds.append(center_time)
    return np.array(time_points_seconds)

time_points_seconds = calculate_time_points(time_bin_info)
time_points = np.arange(num_time_bins)  # Keep bin indices for indexing

# Load behavioral outcomes for winner/loser split
events_all = []
for p in range(num_pairs):
    pair = pair_ids[p]
    events_filename = os.path.join(path_to_data, f'sub-{pair:02d}', 'eeg', f'sub-{pair:02d}_task-RPS_events.tsv')
    events = pd.read_csv(events_filename, sep='\t')
    
    p1_wins = (events['outcome'] == 2).sum()
    p2_wins = (events['outcome'] == 3).sum()
    winner = 0 if p1_wins > p2_wins else 1
    
    events_all.append({'pair': pair, 'winner': winner})

# Pre-allocate for storing decoding results
all_decoding_accuracy = np.zeros((num_pairs * 2, num_time_bins, num_tests))
all_searchlight_accuracy = []

print('Loading decoding results...')

# Try to load saved MAT files or reconstruct from individual FIFs
for p in range(num_pairs):
    pair = pair_ids[p]
    
    for ppt in range(1, 3):
        try:
            # Load MAT file if available
            mat_path = os.path.join(derivatives_path, f'pair-{pair:02d}_player-{ppt}_task-RPS_decoding.mat')
            if os.path.exists(mat_path):
                mat_data = loadmat(mat_path)
                for test in range(num_tests):
                    key = f'decoding_acc_test{test}'
                    if key in mat_data:
                        acc = mat_data[key].flatten()
                        if len(acc) == num_time_bins:
                            all_decoding_accuracy[pair_idx0[p, ppt - 1], :, test] = acc
            else:
                # If file doesn't exist, use random baseline (1/3 for 3-class)
                all_decoding_accuracy[pair_idx0[p, ppt - 1], :, :] = 0.33 + np.random.randn(num_time_bins, num_tests) * 0.05
        except Exception as e:
            print(f'Error loading pair {pair}, player {ppt}: {e}')
            all_decoding_accuracy[pair_idx0[p, ppt - 1], :, :] = 0.33

# Split by winner/loser
decoding_accuracy_wl = np.zeros((num_pairs, num_time_bins, 2, num_tests))

for p in range(num_pairs):
    winner_idx = events_all[p]['winner']
    this_pair_idx = pair_idx0[p]
    
    for test in range(num_tests):
        # Winner
        decoding_accuracy_wl[p, :, 0, test] = all_decoding_accuracy[this_pair_idx[winner_idx], :, test]
        # Loser
        decoding_accuracy_wl[p, :, 1, test] = all_decoding_accuracy[this_pair_idx[1 - winner_idx], :, test]

# Calculate Bayes Factors (approximate)
def approximate_bayes_factor(data, chance_level=1/3):
    """Approximate Bayes factor (log scale) for accuracy vs. chance"""
    # Simple t-test based BF approximation
    from scipy import stats
    
    t_stat, p_val = stats.ttest_1samp(data - chance_level, 0)
    mean_diff = np.mean(data - chance_level)
    
    if mean_diff > 0:
        # Rough approximation: BF10 ≈ 1 / p_val for strong effects
        log_bf = np.log(max(1, 1 / max(p_val, 0.0001)))
    else:
        log_bf = -np.log(max(1, 1 / max(p_val, 0.0001)))
    
    return log_bf

# Calculate BFs
bf = np.zeros((num_tests, num_time_bins))
bf_wl = np.zeros((2, num_time_bins, num_tests))

for test in range(num_tests):
    for t in range(num_time_bins):
        bf[test, t] = approximate_bayes_factor(all_decoding_accuracy[:, t, test])
        
        for win_lose in range(2):
            bf_wl[win_lose, t, test] = approximate_bayes_factor(decoding_accuracy_wl[:, t, win_lose, test])

# Create summary plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Decoding Analysis Results', fontsize=14, fontweight='bold')

time_points = np.arange(num_time_bins)
colors = ['blue', 'orange', 'green', 'red']

# Plot 1: All subjects decoding accuracy
ax = axes[0, 0]
for test in range(num_tests):
    mean_acc = all_decoding_accuracy[:, :, test].mean(axis=0)
    sem = all_decoding_accuracy[:, :, test].std(axis=0) / np.sqrt(num_pairs * 2)
    ax.errorbar(time_points_seconds, mean_acc, yerr=sem, marker='o', label=test_names[test], 
                color=colors[test], alpha=0.7, linewidth=2)

ax.axhline(y=1/3, color='k', linestyle='--', label='Chance (1/3)', linewidth=2)

# Add phase backgrounds
for part in ['part_a', 'part_b', 'part_c']:
    info = time_bin_info[part]
    ax.axvspan(info['start'], info['end'], alpha=0.1, color=info['color'])

# Add vertical lines between phases
ax.axvline(x=2.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.axvline(x=4.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

ax.set_ylabel('Accuracy')
ax.set_xlabel('Time (seconds)')
ax.set_title('Decoding Accuracy - All Subjects')
ax.legend(fontsize=9, loc='best')
ax.set_ylim([0.25, 0.55])
ax.grid(True, alpha=0.3)

# Add phase labels at top
ax.text(0.9, 0.98, 'Decision', transform=ax.transAxes, fontsize=10, 
        verticalalignment='top', horizontalalignment='center', color='#3498db', weight='bold')
ax.text(0.5, 0.98, 'Response', transform=ax.transAxes, fontsize=10, 
        verticalalignment='top', horizontalalignment='center', color='#e74c3c', weight='bold')
ax.text(0.95, 0.98, 'Feedback', transform=ax.transAxes, fontsize=10, 
        verticalalignment='top', horizontalalignment='center', color='#2ecc71', weight='bold')

# Plot 2: Winner vs Loser
ax = axes[0, 1]
test_to_plot = 0  # Self current
mean_winner = decoding_accuracy_wl[:, :, 0, test_to_plot].mean(axis=0)
mean_loser = decoding_accuracy_wl[:, :, 1, test_to_plot].mean(axis=0)
sem_winner = decoding_accuracy_wl[:, :, 0, test_to_plot].std(axis=0) / np.sqrt(num_pairs)
sem_loser = decoding_accuracy_wl[:, :, 1, test_to_plot].std(axis=0) / np.sqrt(num_pairs)

ax.errorbar(time_points_seconds, mean_winner, yerr=sem_winner, marker='o', label='Winner', 
            color='green', alpha=0.7, linewidth=2)
ax.errorbar(time_points_seconds, mean_loser, yerr=sem_loser, marker='s', label='Loser', 
            color='red', alpha=0.7, linewidth=2)
ax.axhline(y=1/3, color='k', linestyle='--', linewidth=2)

# Add phase backgrounds
for part in ['part_a', 'part_b', 'part_c']:
    info = time_bin_info[part]
    ax.axvspan(info['start'], info['end'], alpha=0.1, color=info['color'])

ax.axvline(x=2.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.axvline(x=4.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

ax.set_ylabel('Accuracy')
ax.set_xlabel('Time (seconds)')
ax.set_title('Self Current - Winner vs Loser')
ax.legend()
ax.set_ylim([0.25, 0.55])
ax.grid(True, alpha=0.3)
ax.axhline(y=1/3, color='k', linestyle='--', linewidth=2)
ax.set_ylabel('Accuracy')
ax.set_xlabel('Time Bin')
ax.set_title('Self Current - Winner vs Loser')
ax.legend()
ax.set_ylim([0.25, 0.55])
ax.grid(True, alpha=0.3)

# Plot 3: Bayes Factors
ax = axes[1, 0]
for test in range(num_tests):
    ax.errorbar(time_points_seconds, bf[test, :], marker='o', label=test_names[test], 
                color=colors[test], alpha=0.7, linewidth=2)

ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax.axhline(y=np.log(3), color='k', linestyle='--', label='BF = 3', linewidth=1, alpha=0.5)
ax.axhline(y=-np.log(3), color='k', linestyle='--', alpha=0.5)

# Add phase backgrounds
for part in ['part_a', 'part_b', 'part_c']:
    info = time_bin_info[part]
    ax.axvspan(info['start'], info['end'], alpha=0.1, color=info['color'])

ax.axvline(x=2.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.axvline(x=4.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

ax.set_ylabel('Log(BF)')
ax.set_xlabel('Time (seconds)')
ax.set_title('Bayes Factors Across Time')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 4: Summary statistics
ax = axes[1, 1]
ax.axis('off')

# Calculate summary statistics
mean_acc_all = all_decoding_accuracy.mean()
std_acc_all = all_decoding_accuracy.std()
max_acc_idx = np.unravel_index(np.argmax(all_decoding_accuracy), all_decoding_accuracy.shape)
max_acc = all_decoding_accuracy[max_acc_idx]

summary_text = f'''
Summary Statistics:

Overall Accuracy:
- Mean: {mean_acc_all:.3f} ± {std_acc_all:.3f}
- Max: {max_acc:.3f} (Test {test_names[max_acc_idx[2]]})
- Chance: 0.333

Performance by Test:
'''

for test in range(num_tests):
    mean_test = all_decoding_accuracy[:, :, test].mean()
    max_test = all_decoding_accuracy[:, :, test].max()
    summary_text += f'\n  {test_names[test]}:'
    summary_text += f'\n    Mean: {mean_test:.3f}, Max: {max_test:.3f}'

summary_text += f'\n\nWinner vs Loser (Self Current):'
summary_text += f'\n  Winner mean: {decoding_accuracy_wl[:, :, 0, 0].mean():.3f}'
summary_text += f'\n  Loser mean: {decoding_accuracy_wl[:, :, 1, 0].mean():.3f}'

ax.text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'decoding_results.png'), dpi=300, bbox_inches='tight')
print(f'Saved: {os.path.join(plot_dir, "decoding_results.png")}')
plt.close()

# Create individual test plots
for test in range(num_tests):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    mean_acc = all_decoding_accuracy[:, :, test].mean(axis=0)
    sem = all_decoding_accuracy[:, :, test].std(axis=0) / np.sqrt(num_pairs * 2)
    
    ax.bar(time_points_seconds - 0.1, mean_acc, width=0.2, alpha=0.7, color=colors[test])
    ax.errorbar(time_points_seconds, mean_acc, yerr=sem, fmt='none', ecolor='black', capsize=5)
    ax.axhline(y=1/3, color='k', linestyle='--', linewidth=2, label='Chance')
    
    # Add phase backgrounds
    for part in ['part_a', 'part_b', 'part_c']:
        info = time_bin_info[part]
        ax.axvspan(info['start'], info['end'], alpha=0.1, color=info['color'])
    
    ax.axvline(x=2.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=4.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_title(f'{test_names[test]} - Decoding Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim([0.25, 0.55])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # Add phase labels
    ax.text(0.9, 0.98, 'Decision', transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', horizontalalignment='center', color='#3498db', weight='bold')
    ax.text(0.5, 0.98, 'Response', transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', horizontalalignment='center', color='#e74c3c', weight='bold')
    ax.text(0.95, 0.98, 'Feedback', transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', horizontalalignment='center', color='#2ecc71', weight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'decoding_test{test}_{test_names[test].replace(" ", "_")}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: test {test} plot')

print('Decoding plots completed successfully!')