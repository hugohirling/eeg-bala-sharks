"""
Behavioral Analysis Module
- Markov chain model for predictability
- Win-Stay, Lose-Shift, Cycling detection
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add the root directory to sys.path to import paths.py
# If this file is in eeg_pipeline/analysis/, root is 2 directories up
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import paths


# =============================================================================
# MARKOV CHAIN ANALYSIS
# =============================================================================

def compute_markov_transition_matrix(choices):
    """
    Compute transition probabilities between consecutive choices.
    
    Parameters
    ----------
    choices : array-like
        Sequence of choices (0=Rock, 1=Paper, 2=Scissors)
    
    Returns
    -------
    transition_matrix : np.ndarray
        3x3 matrix of P(next_choice | current_choice)
    """
    counts = np.zeros((3, 3))
    for i in range(len(choices) - 1):
        counts[choices[i], choices[i + 1]] += 1
    
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums


def compute_predictability(matrix):
    """
    Compute predictability score from transition matrix.
    
    Returns
    -------
    predictability : float
        Score from 0 (random) to 1 (fully predictable)
    """
    max_entropy = np.log2(3)
    entropies = []
    
    for row in matrix:
        if row.sum() > 0:
            row = row[row > 0]
            entropies.append(-np.sum(row * np.log2(row)))
    
    avg_entropy = np.mean(entropies) if entropies else max_entropy
    return 1 - (avg_entropy / max_entropy)


# =============================================================================
# STRATEGY CLASSIFICATION
# =============================================================================

def classify_strategy(choices, outcomes):
    """
    Classify player strategy (Win-Stay, Lose-Shift).
    
    Parameters
    ----------
    choices : array-like
        Sequence of choices (0=Rock, 1=Paper, 2=Scissors)
    outcomes : array-like
        Sequence of outcomes (1=win, 0=draw, -1=loss)
    
    Returns
    -------
    strategies : dict
        Dictionary with strategy rates
    """
    win_stay = lose_shift = win_opp = lose_opp = 0
    
    for i in range(len(choices) - 1):
        if outcomes[i] == 1:
            win_opp += 1
            if choices[i + 1] == choices[i]:
                win_stay += 1
        elif outcomes[i] == -1:
            lose_opp += 1
            if choices[i + 1] != choices[i]:
                lose_shift += 1
    
    return {
        'win_stay_rate': win_stay / win_opp if win_opp > 0 else 0,
        'lose_shift_rate': lose_shift / lose_opp if lose_opp > 0 else 0,
        'win_opportunities': win_opp,
        'lose_opportunities': lose_opp
    }


def detect_cycling(choices):
    """
    Detect cycling patterns (R->P->S->R or R->S->P->R).
    
    Returns
    -------
    cycling : dict
        Dictionary with cycling rates
    """
    forward_cycle = 0
    backward_cycle = 0
    total_transitions = len(choices) - 1
    
    for i in range(len(choices) - 1):
        current = choices[i]
        next_choice = choices[i + 1]
        
        if (current + 1) % 3 == next_choice:
            forward_cycle += 1
        elif (current - 1) % 3 == next_choice:
            backward_cycle += 1
    
    return {
        'forward_cycle_rate': forward_cycle / total_transitions,
        'backward_cycle_rate': backward_cycle / total_transitions,
        'total_cycle_rate': (forward_cycle + backward_cycle) / total_transitions,
        'random_expected': 2/3
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_behavioral_data(events_file):
    """
    Load behavioral data from BIDS events file.
    
    Parameters
    ----------
    events_file : str
        Path to events.tsv file
    
    Returns
    -------
    data : dict
        Dictionary with player choices and outcomes
    """
    events_df = pd.read_csv(events_file, sep='\t')
    
    player1_choices = (events_df['player1_resp'] - 1).tolist()
    player2_choices = (events_df['player2_resp'] - 1).tolist()
    
    outcomes = events_df['outcome'].tolist()
    player1_outcomes = [0 if o == 1 else (1 if o == 2 else -1) for o in outcomes]
    player2_outcomes = [0 if o == 1 else (1 if o == 3 else -1) for o in outcomes]
    
    return {
        'player1_choices': player1_choices,
        'player2_choices': player2_choices,
        'player1_outcomes': player1_outcomes,
        'player2_outcomes': player2_outcomes,
        'n_trials': len(events_df)
    }


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_player(choices, outcomes, player_name="Player"):
    """
    Run complete behavioral analysis for one player.
    
    Returns
    -------
    results : dict
        Complete analysis results
    """
    matrix = compute_markov_transition_matrix(choices)
    predictability = compute_predictability(matrix)
    strategies = classify_strategy(choices, outcomes)
    cycling = detect_cycling(choices)
    
    return {
        'transition_matrix': matrix,
        'predictability': predictability,
        'win_stay_rate': strategies['win_stay_rate'],
        'lose_shift_rate': strategies['lose_shift_rate'],
        'cycling_rate': cycling['total_cycle_rate'],
        'forward_cycling': cycling['forward_cycle_rate'],
        'backward_cycling': cycling['backward_cycle_rate']
    }
    

def process_subject(subject_id: str):
    """
    Run behavioral analysis for a specific subject.
    """
    print(f"\n[{subject_id}] Starting behavioral analysis...")
    
    # Setup Inputs and Outputs
    # Behavioral data is usually derived from the original BIDS events.tsv file
    bids_events_file = paths.INPUT_DIR / f"sub-{subject_id}" / "eeg" / f"sub-{subject_id}_task-RPS_events.tsv"
    
    output_dir = paths.OUTPUT_DIR / "analysis" / "behavioral"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not bids_events_file.exists():
        print(f"[{subject_id}] WARNING: Events file not found at {bids_events_file}. Skipping.")
        return

    print(f"[{subject_id}] Loading behavioral data from: {bids_events_file.name}")
    
    try:
        # Assuming load_behavioral_data extracts the choices/outcomes properly
        data = load_behavioral_data(bids_events_file)
        
        print(f"[{subject_id}] Running Markov Models and Strategy Analysis...")
        p1_results = analyze_player(data['player1_choices'], data['player1_outcomes'], "Player 1")
        p2_results = analyze_player(data['player2_choices'], data['player2_outcomes'], "Player 2")
        
        # 1. Save output matrices (.npy)
        p1_out_path = output_dir / f"sub-{subject_id}_P1_markov_matrix.npy"
        p2_out_path = output_dir / f"sub-{subject_id}_P2_markov_matrix.npy"
        
        np.save(p1_out_path, p1_results['transition_matrix'])
        np.save(p2_out_path, p2_results['transition_matrix'])
        
        # 2. Save the other scalar metrics (.json)
        # We create a copy and remove the matrix since it can't be saved in standard JSON easily
        import json
        
        p1_metrics = {k: v for k, v in p1_results.items() if k != 'transition_matrix'}
        p2_metrics = {k: v for k, v in p2_results.items() if k != 'transition_matrix'}
        
        with open(output_dir / f"sub-{subject_id}_P1_metrics.json", "w") as f:
            json.dump(p1_metrics, f, indent=4)
            
        with open(output_dir / f"sub-{subject_id}_P2_metrics.json", "w") as f:
            json.dump(p2_metrics, f, indent=4)
        
        print(f"[{subject_id}] SUCCESS: Saved matrices and metrics to {output_dir}")
        
    except Exception as e:
        print(f"[{subject_id}] ERROR: Failed during analysis -> {e}")

    print(f"[{subject_id}] Finished behavioral analysis.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run behavioral analysis step.")
    parser.add_argument("--subject", type=str, help="Subject ID (e.g., 01)")
    args = parser.parse_args()
    
    if args.subject:
        process_subject(args.subject)
    else:
        # If no subject is passed, prompt the user
        print("Please provide a subject ID. Example:")
        print("python eeg_pipeline/analysis/07_behavioral_analysis.py --subject 01")