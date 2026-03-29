"""
Plot behavioral responses (Python version of step3a_plot_Fig1.m):
   - Plot response distributions
   - Plot outcomes
   - Analyze 'stay' responses
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Set the path
path_to_data = 'MNE-sample-data/ds006761'
plot_dir = os.path.join(path_to_data, 'derivatives', 'plots')
os.makedirs(plot_dir, exist_ok=True)

# Set parameters
pair_ids = np.concatenate([np.arange(1, 10), np.arange(11, 23), np.arange(25, 35)])
num_pairs = len(pair_ids)
num_blocks = 12
num_trials_per_block = 40
num_trials = num_blocks * num_trials_per_block
response_names = ['Rock', 'Paper', 'Scissors']
outcome_names = ['Draw', 'Winner', 'Loser']

# Pre-allocate output
outcome_summary = np.zeros((num_pairs, 3))
all_played_rank = np.zeros((3, num_pairs * 2))
ranked_resp = np.zeros((3, num_pairs * 2))
prop_stay = np.zeros((3, num_pairs * 2))

# Reshape pair index (1-based for display, use 0-based for array indexing)
pair_idx = np.arange(1, num_pairs * 2 + 1).reshape(-1, 2)
pair_idx0 = pair_idx - 1  # zero-based indices for Python arrays

fig_count = 0

# Loop over pairs
for p in range(num_pairs):
    pair = pair_ids[p]
    print(f'Analyzing pair {p + 1} of {num_pairs}: sub-{pair:02d}')

    # Load behavioral events
    events_filename = os.path.join(path_to_data, f'sub-{pair:02d}', 'eeg', f'sub-{pair:02d}_task-RPS_events.tsv')
    events = pd.read_csv(events_filename, sep='\t')

    # Determine winner
    p1_wins = (events['outcome'] == 2).sum()
    p2_wins = (events['outcome'] == 3).sum()
    draws = (events['outcome'] == 1).sum()

    if p1_wins > p2_wins:
        winner_idx = 0
    elif p2_wins > p1_wins:
        winner_idx = 1
    else:
        winner_idx = 2

    # Remove trials with no response
    events_r = events[(events['player1_resp'] > 0) & (events['player2_resp'] > 0)]

    # Get outcome summary
    outcome_summary[p, 0] = (events_r['outcome'] == 1).sum() / len(events_r) * 100  # Draw
    if winner_idx < 2:
        outcome_summary[p, 1] = (events_r['outcome'] == (winner_idx + 2)).sum() / len(events_r) * 100  # Winner
        outcome_summary[p, 2] = (events_r['outcome'] != (winner_idx + 2)).sum() / len(events_r) * 100  # Loser
    else:
        outcome_summary[p, 1:] = 50

    # Get response frequencies
    all_played = np.zeros((3, 2))
    played = np.column_stack([events_r['player1_resp'], events_r['player2_resp']])
    
    for ppt in range(2):
        for resp in range(1, 4):
            all_played[resp - 1, ppt] = (played[:, ppt] == resp).sum()
        all_played[:, ppt] = all_played[:, ppt] / played.shape[0] * 100

        # Rank responses
        rank_idx = np.argsort(-all_played[:, ppt])
        subj_idx = pair_idx0[p, ppt]
        all_played_rank[:, subj_idx] = rank_idx + 1

    # Get ranked response frequencies
    ranked = np.sort(all_played)[::-1]
    ranked_resp[:, pair_idx0[p, 0]] = ranked[:, 0]
    ranked_resp[:, pair_idx0[p, 1]] = ranked[:, 1]

    # Calculate "stay" responses (same response twice in a row)
    Player_1_Behav = events[['player1_resp', 'player2_resp', 'outcome']].values.copy()
    Player_2_Behav = events[['player2_resp', 'player1_resp', 'outcome']].values.copy()

    # Fix outcome coding for player 2
    Player_2_Behav[events['outcome'] == 1, 2] = 1
    Player_2_Behav[events['outcome'] == 2, 2] = 3
    Player_2_Behav[events['outcome'] == 3, 2] = 2

    # Reshape into blocks
    Player_1_Behav = Player_1_Behav.reshape(num_blocks, num_trials_per_block, 3)
    Player_2_Behav = Player_2_Behav.reshape(num_blocks, num_trials_per_block, 3)

    # Calculate stay responses per outcome
    for ppt_behav, ppt_idx in [(Player_1_Behav, 0), (Player_2_Behav, 1)]:
        for outcome_type in range(1, 4):  # Draw, Win, Lose
            stay_count = 0
            total_count = 0

            for block in range(num_blocks):
                for trial in range(1, num_trials_per_block):
                    if ppt_behav[block, trial - 1, 2] == outcome_type and ppt_behav[block, trial - 1, 0] > 0:
                        total_count += 1
                        if ppt_behav[block, trial, 0] == ppt_behav[block, trial - 1, 0]:
                            stay_count += 1

            if total_count > 0:
                prop_stay[outcome_type - 1, pair_idx0[p, ppt_idx]] = stay_count / total_count

# Create summary plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Behavioral Summary Across Pairs', fontsize=14, fontweight='bold')

# Plot 1: Outcome summary
ax = axes[0, 0]
x = np.arange(num_pairs)
width = 0.25
ax.bar(x - width, outcome_summary[:, 0], width, label='Draw', alpha=0.8)
ax.bar(x, outcome_summary[:, 1], width, label='Winner', alpha=0.8)
ax.bar(x + width, outcome_summary[:, 2], width, label='Loser', alpha=0.8)
ax.set_ylabel('Percentage (%)')
ax.set_xlabel('Pair')
ax.set_title('Outcome Distribution')
ax.legend()
ax.set_ylim([0, 100])

# Plot 2: Response distribution
ax = axes[0, 1]
for i in range(3):
    ax.plot(ranked_resp[i, :], marker='o', label=response_names[i], alpha=0.7)
ax.set_ylabel('Percentage (%)')
ax.set_xlabel('Subject ID')
ax.set_title('Response Frequency Ranked')
ax.legend()

# Plot 3: Stay responses by outcome
ax = axes[1, 0]
for outcome in range(3):
    ax.plot(prop_stay[outcome, :], marker='o', label=outcome_names[outcome], alpha=0.7)
ax.set_ylabel('Proportion Stay')
ax.set_xlabel('Subject ID')
ax.set_title('Stay Proportion by Game Outcome')
ax.set_ylim([0, 1])
ax.legend()

# Plot 4: Summary statistics
ax = axes[1, 1]
ax.axis('off')
summary_text = f'''
Summary Statistics:
- Total pairs: {num_pairs}
- Total subjects: {num_pairs * 2}
- Trials per subject: {num_trials}

Mean Outcomes:
- Draw: {outcome_summary[:, 0].mean():.1f}%
- Winner: {outcome_summary[:, 1].mean():.1f}%
- Loser: {outcome_summary[:, 2].mean():.1f}%

Mean Stay Responses:
- After Draw: {prop_stay[0, :].mean():.3f}
- After Win: {prop_stay[1, :].mean():.3f}
- After Loss: {prop_stay[2, :].mean():.3f}
'''
ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center', family='monospace')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'behavioral_summary.png'), dpi=300, bbox_inches='tight')
print(f'Saved: {os.path.join(plot_dir, "behavioral_summary.png")}')
plt.close()

print('Behavioral plots completed!')