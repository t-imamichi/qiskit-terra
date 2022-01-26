# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Expectation value with Richardson extrapolation
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Union

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.pulse import InstructionScheduleMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.result import Result

from ..results import EstimatorResult
from ..results.base_result import BaseResult
from .base_estimator import BaseEstimator


@dataclass(frozen=True)
class RichardsonSetting:
    """Data class for richardson extrapolation."""

    scales: list[float]
    inst_maps: list[InstructionScheduleMap]


class RichardsonExtrapolationEstimator(BaseEstimator):
    """
    Expectation Value class with Richardson extrapolation
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        richardson_setting: RichardsonSetting,
    ):
        """ """
        super().__init__(
            estimator.circuit,
            estimator.observable,
            sampler=estimator._backend,
        )
        self._expectation_value = copy.copy(estimator)
        if not isinstance(richardson_setting, RichardsonSetting):
            richardson_setting = RichardsonSetting(**richardson_setting)
        self._richardson_setting = richardson_setting

    @property
    def transpiled_circuits(self):
        if self._transpiled_circuits is None:
            transpiled_circuits = []
            for inst_map in self._richardson_setting.inst_maps:
                transpile_opts = copy.copy(self.transpile_options)
                transpile_opts.update_options(inst_map=inst_map)
                self._expectation_value.set_transpile_options(**transpile_opts.__dict__)
                transpiled_circuits.append(self._expectation_value.transpiled_circuits)
            self._transpiled_circuits = sum(transpiled_circuits, [])
        return self._transpiled_circuits

    def _postprocessing(self, result: Union[Result, BaseResult, dict]) -> EstimatorResult:
        expval_length = len(self._expectation_value.transpiled_circuits)
        scales = self._richardson_setting.scales
        c_mat = np.array([[scale ** i for scale in scales] for i in range(len(scales))])
        vec = np.array([1] + [0] * (len(scales) - 1))
        coeffs = np.linalg.solve(c_mat, vec)
        value = 0
        raw_results = []
        for i, (coeff, scale) in enumerate(zip(coeffs, scales)):
            expval_result = self._expectation_value._postprocessing(
                result[i * expval_length : (i + 1) * expval_length]  # type: ignore
            )
            raw_results.append({"expectation_value_result": expval_result, "scale": scale})
            value += coeff * expval_result.value  # type: ignore
        return EstimatorResult(value, raw_data={"raw_results": raw_results})

    def _preprocessing(
        self, circuit: QuantumCircuit, observable: SparsePauliOp
    ) -> Union[list[QuantumCircuit], tuple[QuantumCircuit, list[QuantumCircuit]]]:
        return self._expectation_value.preprocessed_circuits
