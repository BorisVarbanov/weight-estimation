from numba import float64, vectorize


@vectorize([float64(float64, float64)])
def add_probs(prob_i: float, prob_j: float) -> float:
    """
    combine_events Calculates the joint probability of two
    independent/uncorrelated error events happening with probabilities
    prob_i and prob_j respectively.

    Parameters
    ----------
    prob_i : float
        The probability of the first error event.
    prob_j : float
        The probability of the second error event.

    Returns
    -------
    float
        The joint probability of the two error events.
    """
    return prob_i + prob_j - (2 * prob_i * prob_j)
