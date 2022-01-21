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
Expectation value class
"""

# pylint: disable=no-name-in-module, import-error

from __future__ import annotations

from typing import Union

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.opflow import PauliSumOp
from qiskit.providers import BackendV1 as Backend
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result import Result
from qiskit.utils import has_aer

from ..results import EstimatorArrayResult
from ..results.base_result import BaseResult
from .base_estimator import BaseEstimator

if has_aer():
    from qiskit.providers.aer.library import SaveExpectationValueVariance


class ExactEstimator(BaseEstimator):
    """
    Calculates the expectation value exactly (i.e. without sampling error).
    """

    def __init__(
        self,
        circuits: list[Union[QuantumCircuit, Statevector]],
        observables: list[Union[BaseOperator, PauliSumOp]],
        backend: Backend,
    ):
        if not has_aer():
            raise MissingOptionalLibraryError(
                libname="qiskit-aer",
                name="Aer provider",
                pip_install="pip install qiskit-aer",
            )

        super().__init__(
            circuits=circuits,
            observables=observables,
            sampler=backend,
        )

    def _preprocessing(
        self, circuits: list[QuantumCircuit], observables: list[SparsePauliOp]
    ) -> list[QuantumCircuit]:
        preprocessed_circuits = []
        for group in self._grouping:
            circuit_copy = circuits[group.circuit_index].copy()
            circuit_copy.append(
                SaveExpectationValueVariance(operator=observables[group.observable_index]),
                qargs=range(circuit_copy.num_qubits),
            )
            preprocessed_circuits.append(circuit_copy)
        return preprocessed_circuits

    def _postprocessing(self, result: Union[Result, BaseResult, dict]) -> EstimatorArrayResult:

        # TODO: validate

        expvals = []
        variances = []
        for i, _ in enumerate(self._grouping):
            if isinstance(result, Result):
                expval, variance = result.data(i)["expectation_value_variance"]
            else:
                # TODO: Fix following type ignore
                expval, variance = result[i].to_dict()["data"][  # type: ignore
                    "expectation_value_variance"
                ]
            expvals.append(expval)
            variances.append(variance)
        return EstimatorArrayResult(
            np.array(expvals, dtype=np.float64), np.array(variances, dtype=np.float64)
        )
