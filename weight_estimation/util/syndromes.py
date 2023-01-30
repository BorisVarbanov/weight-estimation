from typing import Optional

from xarray import DataArray


def get_syndromes(anc_meas: DataArray) -> DataArray:
    syndromes = anc_meas ^ anc_meas.shift(qec_round=1, fill_value=0)
    return syndromes


def get_defects(syndromes: DataArray, frame: Optional[DataArray] = None) -> DataArray:
    shifted_syn = syndromes.shift(qec_round=1, fill_value=0)

    if frame is not None:
        shifted_syn[dict(qec_round=0)] = frame

    defects = syndromes ^ shifted_syn
    return defects
