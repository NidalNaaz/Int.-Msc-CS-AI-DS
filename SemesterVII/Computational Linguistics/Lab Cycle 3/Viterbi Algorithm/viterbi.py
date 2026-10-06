def viterbi(words, states, transition_prob, emission_prob, start_state="START", stop_state="STOP"):
    """
    Implements the Viterbi algorithm for Part-of-Speech tagging.
    """
    V = [{}]
    path = {}

    # Step 1: Initialization
    for state in states:
        trans_p = transition_prob.get(start_state, {}).get(state, 0.0)
        emiss_p = emission_prob.get(state, {}).get(words[0], 0.0)
        V[0][state] = trans_p * emiss_p
        path[state] = [state]

    # Step 2: Recursion over remaining words
    for t in range(1, len(words)):
        V.append({})
        new_path = {}

        for curr_state in states:
            # Find the best prior state leading to current_state
            (max_prob, best_prev_state) = max(
                (
                    V[t - 1][prev_state]
                    * transition_prob.get(prev_state, {}).get(curr_state, 0.0)
                    * emission_prob.get(curr_state, {}).get(words[t], 0.0),
                    prev_state,
                )
                for prev_state in states
            )

            V[t][curr_state] = max_prob
            new_path[curr_state] = path[best_prev_state] + [curr_state]

        path = new_path

    # Step 3: Termination (Transition to STOP state)
    (max_final_prob, best_last_state) = max(
        (
            V[len(words) - 1][last_state]
            * transition_prob.get(last_state, {}).get(stop_state, 0.0),
            last_state,
        )
        for last_state in states
    )

    best_sequence = path[best_last_state]
    return best_sequence, max_final_prob


if __name__ == "__main__":
    # States (POS tags)
    states = ["NN", "VB", "JJ", "RB"]

    # Transition Probabilities (a_ij)[cite: 1]
    transition_prob = {
        "START": {"STOP": 0.0, "NN": 0.5, "VB": 0.25, "JJ": 0.25, "RB": 0.0},
        "NN": {"STOP": 0.25, "NN": 0.25, "VB": 0.5, "JJ": 0.0, "RB": 0.0},
        "VB": {"STOP": 0.25, "NN": 0.25, "VB": 0.0, "JJ": 0.25, "RB": 0.25},
        "JJ": {"STOP": 0.0, "NN": 0.75, "VB": 0.0, "JJ": 0.25, "RB": 0.0},
        "RB": {"STOP": 0.5, "NN": 0.25, "VB": 0.0, "JJ": 0.25, "RB": 0.0},
    }

    # Emission Probabilities (b_ik)[cite: 1]
    emission_prob = {
        "NN": {"time": 0.1, "flies": 0.01, "fast": 0.01},
        "VB": {"time": 0.01, "flies": 0.1, "fast": 0.01},
        "JJ": {"time": 0.0, "flies": 0.0, "fast": 0.1},
        "RB": {"time": 0.0, "flies": 0.0, "fast": 0.1},
    }

    # Input Sentence
    sentence = ["time", "flies", "fast"]

    # Run Viterbi algorithm
    best_tags, probability = viterbi(sentence, states, transition_prob, emission_prob)

    # Print results
    print("=" * 45)
    print(" VITERBI POS TAGGER OUTPUT")
    print("=" * 45)
    print(f"Input Sentence      : {' '.join(sentence)}")
    print(f"Most Probable Tags  : {' -> '.join(best_tags)}")
    print(f"Sequence Probability: {probability:.8f}")
    print("=" * 45)
