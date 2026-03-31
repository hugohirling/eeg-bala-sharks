"""
Behavioral Analysis Module

This module analyzes the behavioral data from the Rock-Paper-Scissors (RPS) task.
It extracts cognitive strategies and decision-making heuristics used by the players,
including Markov chain models for transition predictability, Shannon entropy for 
randomness scoring, and Win-Stay/Lose-Shift (WSLS) operant conditioning metrics.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure the root directory is accessible to import custom path configurations
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import paths


# =============================================================================
# MARKOV CHAIN ANALYSIS
# =============================================================================

def compute_markov_transition_matrix(choices):
    """
    Compute transition probabilities between consecutive game choices.
    
    Generates a 3x3 heatmap representing the probability that a player will 
    choose a specific action given their immediately preceding action.
    
    Parameters
    ----------
    choices : list or np.ndarray
        Sequence of historical choices (0=Rock, 1=Paper, 2=Scissors).
    
    Returns
    -------
    transition_matrix : np.ndarray
        3x3 matrix representing P(next_choice | current_choice).
    """
    # Initialize an empty 3x3 matrix for transitions (R->R, R->P, etc.)
    counts = np.zeros((3, 3))
    
    # Tally the actual transitions from trial to trial
    for i in range(len(choices) - 1):
        counts[choices[i], choices[i + 1]] += 1
    
    # Normalize rows to create probabilities summing to 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    # Prevent division by zero for states that were never chosen
    row_sums[row_sums == 0] = 1 
    
    return counts / row_sums


def compute_predictability(matrix):
    """
    Quantifies how predictable a player is based on their transition matrix 
    using normalized Shannon entropy.
    
    Parameters
    ----------
    matrix : np.ndarray
        3x3 Markov transition matrix.
        
    Returns
    -------
    predictability : float
        Score bounded between 0.0 (perfectly random/unpredictable) and 
        1.0 (highly repetitive/fully predictable).
    """
    # Max entropy for 3 choices is log2(3) -> ~1.58 bits
    max_entropy = np.log2(3)
    entropies = []
    
    # Calculate row-wise Shannon entropy for valid transition distributions
    for row in matrix:
        if row.sum() > 0:
            row = row[row > 0] # Filter out zeros to avoid log2(0) errors
            entropies.append(-np.sum(row * np.log2(row)))
    
    # Average the entropy across all rows
    avg_entropy = np.mean(entropies) if entropies else max_entropy
    
    # Invert entropy to yield a 'predictability' score
    return 1 - (avg_entropy / max_entropy)


# =============================================================================
# STRATEGY CLASSIFICATION
# =============================================================================

def classify_strategy(choices, outcomes):
    """
    Classify player strategy based on standard reinforcement learning heuristics:
    - Win-Stay: Repeating the same action after a win.
    - Lose-Shift: Changing action after a loss.
    
    Parameters
    ----------
    choices : list or np.ndarray
        Sequence of choices (0=Rock, 1=Paper, 2=Scissors).
    outcomes : list or np.ndarray
        Sequence of trial outcomes (1=win, 0=draw, -1=loss).
    
    Returns
    -------
    strategies : dict
        Calculated rates mapping player conditioning logic.
    """
    win_stay = lose_shift = win_opp = lose_opp = 0
    
    for i in range(len(choices) - 1):
        # Case 1: Player won the current trial
        if outcomes[i] == 1:
            win_opp += 1
            if choices[i + 1] == choices[i]: # Did they repeat the move?
                win_stay += 1
        # Case 2: Player lost the current trial
        elif outcomes[i] == -1:
            lose_opp += 1
            if choices[i + 1] != choices[i]: # Did they change their move?
                lose_shift += 1
    
    # Safely compute percentages
    return {
        'win_stay_rate': win_stay / win_opp if win_opp > 0 else 0,
        'lose_shift_rate': lose_shift / lose_opp if lose_opp > 0 else 0,
        'win_opportunities': win_opp,
        'lose_opportunities': lose_opp
    }


def detect_cycling(choices):
    """
    Detect explicit sequences and patterns in decision making. 
    Forward cycling: Rock->Paper->Scissors
    Backward cycling: Rock->Scissors->Paper
    
    Parameters
    ----------
    choices : list or np.ndarray
        Sequence of choices (0=Rock, 1=Paper, 2=Scissors).
        
    Returns
    -------
    cycling : dict
        Calculated rates mapping sequential cycling behavior.
    """
    forward_cycle = backward_cycle = 0
    total_transitions = len(choices) - 1
    
    for i in range(len(choices) - 1):
        current = choices[i]
        next_choice = choices[i + 1]
        
        # Use modulo arithmetic to track forward (n+1) and backward (n-1) steps
        if (current + 1) % 3 == next_choice:
            forward_cycle += 1
        elif (current - 1) % 3 == next_choice:
            backward_cycle += 1
    
    return {
        'forward_cycle_rate': forward_cycle / total_transitions if total_transitions else 0,
        'backward_cycle_rate': backward_cycle / total_transitions if total_transitions else 0,
        'total_cycle_rate': (forward_cycle + backward_cycle) / total_transitions if total_transitions else 0,
        'random_expected': 2/3 # Statistically, changing moves should occur 66.6% of the time purely by chance
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_behavioral_data(events_file):
    """
    Extract and parse behavioral event data directly from BIDS-compliant .tsv files.
    
    Parameters
    ----------
    events_file : str or Path
        Target path to the participant's events.tsv file.
    
    Returns
    -------
    data : dict
        Dictionary containing mapped responses and standardized outcomes.
    """
    events_df = pd.read_csv(events_file, sep='\t')
    
    # Map raw button presses (1,2,3) to 0-indexed logic (0,1,2) for math processing
    player1_choices = (events_df['player1_resp'] - 1).tolist()
    player2_choices = (events_df['player2_resp'] - 1).tolist()
    
    # Map raw outcomes: Original TSV typically uses 1=Tie, 2=P1 Wins, 3=P2 Wins
    # We standardize this to (1=Win, 0=Tie, -1=Loss) relative to each player
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
    Execute the entire cognitive behavioral analysis pipeline for a single player.
    
    Returns
    -------
    results : dict
        Consolidated metrics bundle.
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
    Primary processing function to coordinate subject-level behavioral extraction.
    Exports matrices as NumPy binaries and JSON scalar records for statistical pooling.
    """
    print(f"\n[{subject_id}] Starting behavioral analysis...")
    
    # Define and validate paths
    bids_events_file = paths.INPUT_DIR / f"sub-{subject_id}" / "eeg" / f"sub-{subject_id}_task-RPS_events.tsv"
    
    output_dir = paths.OUTPUT_DIR / "analysis" / "behavioral"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not bids_events_file.exists():
        print(f"[{subject_id}] WARNING: Events file not found at {bids_events_file}. Skipping.")
        return

    print(f"[{subject_id}] Loading behavioral data from: {bids_events_file.name}")
    
    try:
        data = load_behavioral_data(bids_events_file)
        
        print(f"[{subject_id}] Running Markov Models and Strategy Analysis...")
        p1_results = analyze_player(data['player1_choices'], data['player1_outcomes'], "Player 1")
        p2_results = analyze_player(data['player2_choices'], data['player2_outcomes'], "Player 2")
        
        # 1. Save multi-dimensional matrices separately (.npy) for plotting
        p1_out_path = output_dir / f"sub-{subject_id}_P1_markov_matrix.npy"
        p2_out_path = output_dir / f"sub-{subject_id}_P2_markov_matrix.npy"
        
        np.save(p1_out_path, p1_results['transition_matrix'])
        np.save(p2_out_path, p2_results['transition_matrix'])
        
        # 2. Isolate integer/float metrics and save out as JSON strings 
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

# Bootstrap CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run behavioral analysis step.")
    parser.add_argument("--subject", type=str, help="Subject ID (e.g., 01)")
    args = parser.parse_args()
    
    if args.subject:
        process_subject(args.subject)
    else:
        # If no subject is passed, gracefully error and provide guidance
        print("Please provide a subject ID. Example:")
        print("python eeg_pipeline/analysis/07_behavioral_analysis.py --subject 01")