"""
Behavioral Analysis Module
- Markov chain model for predictability
- Win-Stay, Lose-Shift, Cycling detection

Author: Ayush
"""

import numpy as np
import pandas as pd


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
