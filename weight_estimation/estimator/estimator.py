from itertools import product
from typing import Optional

import numpy as np
from pandas import MultiIndex
from xarray import DataArray, Dataset

from .util import add_probs


def get_edge_probs(defects: DataArray, *, normalise: Optional[bool] = False) -> Dataset:
    """
    get_edge_probs Infers the probability of each edges between the possible pairs of defects
    as well as the probability of an edge between each defect and a boundary node, based on the
    observed defect realization provided.

    For the theory behind this formulat see Eqns. 11 and 16 from the article
    "Exponential suppression of bit or phase errors with cyclic error correction" by Google Quantum AI,
    found in the Supplementary information, accessible from https://doi.org/10.1038/s41586-021-03588-y.

    Parameters
    ----------
    defects : DataArray
        The observed defect realization. This variable is a xarray.DataArray that stored the observed
        defect value for each ancilla qubit, each QEC round and each shot of the experiment.

        defects must in this case be a 3D array of binary values observed in experiment.
        The defects.dims must in this base be the list ["shot", "anc_qubit", "qec_round"].
        The coordinate values must be stored in the defects.coords, corresponding to the values each
        array dimension takes at each element of the array.
    normalise : Optional[bool], optional
        Whether to enforce physical edges, such that the probability of each edge is between 0 and 1, by default False.

        This is done by enforcing the value under the square root to be between 0 and 1, such that this estimation
        does not lead to imaginary number (in this code this would leave to nan values) when taking the square root.

    Returns
    -------
    Dataset
        The dataset containing the matrix of edge probabilities and the vector of boundary edge probabalities.
    """
    anc_qubits = defects.anc_qubit.values  # The ancilla qubit labels
    qec_rounds = (
        defects.qec_round.values
    )  # The QEC round number (adopted convetion is to start from round 1).

    nodes = list(
        product(anc_qubits, qec_rounds)
    )  # Each node corresponding to the possible tuples of ancilla labels and round numbers.

    # Converts the 3D xarray.DataArray to 2D numpy.arrays of shots and defect nodes
    # (each node is a tuple of ancilla label and  round number)
    defects_vec = defects.stack(node=("anc_qubit", "qec_round"))
    defects_data = defects_vec.transpose("shot", "node").data

    cross_corr_mat = np.einsum(
        "ni, nj -> nij", defects_data, defects_data
    )  # 3D array of the product of all pairs of defect nodes over each shot
    mean_corr_mat = np.mean(
        cross_corr_mat, axis=0
    )  # The 2D array of the two-defect expectation values

    mean_defs = np.mean(
        defects_data, axis=0
    )  # The 1D array of the one-defect expectation values
    mean_prod_mat = np.einsum(
        "i,j->ij", mean_defs, mean_defs
    )  # 2D array of the possible pair products of all one-defect expectation valuessss
    mean_sum_mat = (
        mean_defs + mean_defs[:, None]
    )  # 2D array of the sums of the possible pairs of one-defect expectation values.

    # The equations below implement Eqn 11 from the reference above.
    numerator = 4 * (mean_corr_mat - mean_prod_mat)
    denominator = 1 - 2 * mean_sum_mat + 4 * mean_corr_mat
    fraction = numerator / denominator

    if normalise:
        fraction[fraction > 1] = 1
        fraction[fraction < 0] = 0

    bulk_probs = 0.5 - 0.5 * np.sqrt(1 - fraction)
    # Main diagonal does not correspond to edge probabilities.
    # Set that to 0 in order to vectorize boundary calculation.
    np.fill_diagonal(bulk_probs, 0)

    # Calculates the probability of each defect to flip due to the edges incident to that defect
    bulk_prob_prods = add_probs.reduce(bulk_probs, axis=0)

    # Implement Eqn. 16 from the reference above.
    bound_probs = (mean_defs - bulk_prob_prods) / (1 - 2 * bulk_prob_prods)

    from_nodes = MultiIndex.from_tuples(
        nodes,
        names=["from_anc_qubit", "from_qec_round"],
    )
    to_nodes = MultiIndex.from_tuples(
        nodes,
        names=["to_anc_qubit", "to_qec_round"],
    )

    error_probs = Dataset(
        data_vars=dict(
            edge=(["from_node", "to_node"], bulk_probs),
            boundary_edge=(["from_node"], bound_probs),
        ),
        coords=dict(from_node=from_nodes, to_node=to_nodes),
    )
    return error_probs
