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

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.opflow import PauliSumOp
from qiskit.providers import BackendV1 as Backend
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result import Result
from qiskit.utils import has_aer

from ..results import EstimatorResult, SamplerResult
from .base_estimator import BaseEstimator

if has_aer():
    from qiskit.providers.aer.library import SaveExpectationValueVariance


class ExactEstimator(BaseEstimator):
    """
    Calculates the expectation value exactly (i.e. without sampling error).
    """

    def __init__(
        self,
        circuit: Union[QuantumCircuit, Statevector],
        observable: Union[BaseOperator, PauliSumOp],
        backend: Backend,
    ):
        if not has_aer():
            raise MissingOptionalLibraryError(
                libname="qiskit-aer",
                name="Aer provider",
                pip_install="pip install qiskit-aer",
            )

        super().__init__(
            circuit=circuit,
            observable=observable,
            backend=backend,
        )

    def _preprocessing(
        self, circuit: QuantumCircuit, observable: SparsePauliOp
    ) -> list[QuantumCircuit]:
        circuit_copy = circuit.copy()
        inst = SaveExpectationValueVariance(operator=observable)
        circuit_copy.append(inst, qargs=range(circuit_copy.num_qubits))
        return [circuit_copy]

    def _postprocessing(self, result: Union[dict, SamplerResult, Result]) -> EstimatorResult:

        # TODO: validate

        if isinstance(result, Result):
            expval, variance = result.data(0)["expectation_value_variance"]
        else:
            expval, variance = result[0].to_dict()["data"]["expectation_value_variance"]
        return EstimatorResult(expval, variance, None)
