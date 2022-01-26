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
Expectation value base class
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union, cast

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.providers import BackendV1 as Backend
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.primitives.framework import BasePrimitive
from qiskit.result import Result

from ..backends import BaseBackendWrapper
from ..results import CompositeResult, EstimatorArrayResult, EstimatorResult
from ..results.base_result import BaseResult
from ..sampler import BaseSampler
from .utils import init_circuit, init_observable

if TYPE_CHECKING:
    from typing import Any


class BaseEstimator(BasePrimitive, ABC):
    """
    Expectation Value class
    """

    def __init__(
        self,
        circuit: Union[QuantumCircuit, Statevector],
        observable: Union[BaseOperator, PauliSumOp],
        sampler: Union[Backend, BaseBackendWrapper, BaseSampler],
    ):
        """ """
        super().__init__(backend=sampler.backend if isinstance(sampler, BaseSampler) else sampler)
        self._circuit = init_circuit(circuit)
        self._observable = init_observable(observable)
        self._sampler = sampler

    @property
    def circuit(self) -> QuantumCircuit:
        """Quantum Circuit that represents quantum state.

        Returns:
            quantum state
        """
        return self._circuit

    @circuit.setter
    def circuit(self, state: Union[QuantumCircuit, Statevector]):
        self._transpiled_circuits = None
        self._circuit = init_circuit(state)

    @property
    def observable(self) -> SparsePauliOp:
        """
        SparsePauliOp that represents observable

        Returns:
            observable
        """
        return self._observable

    @observable.setter
    def observable(self, observable: Union[BaseOperator, PauliSumOp]):
        self._transpiled_circuits = None
        self._observable = init_observable(observable)

    def set_transpile_options(self, **fields) -> BaseEstimator:
        """Set the transpiler options for transpiler.

        Args:
            fields: The fields to update the options
        Returns:
            self
        """
        self._transpiled_circuits = None
        super().set_transpile_options(**fields)
        return self

    @property
    def preprocessed_circuits(
        self,
    ) -> Union[list[QuantumCircuit], tuple[QuantumCircuit, list[QuantumCircuit]]]:
        """
        Transpiled quantum circuits produced by preprocessing

        Returns:
            List of the transpiled quantum circuit
        """
        if self._preprocessed_circuits is None:
            self._preprocessed_circuits = self._preprocessing(self.circuit, self.observable)
        return super().preprocessed_circuits

    def run(
        self,
        parameters: Optional[
            Union[
                list[float],
                list[list[float]],
                "np.ndarray[Any, np.dtype[np.float64]]",
            ]
        ] = None,
        **run_options,
    ) -> Union[EstimatorResult, EstimatorArrayResult]:
        res = super().run(parameters, **run_options)
        if isinstance(res, CompositeResult):
            # TODO CompositeResult should be Generic
            # pylint: disable=no-member
            values = np.array([r.value for r in res.items])  # type: ignore
            variances = np.array([r.variance for r in res.items])  # type: ignore
            confidence_intervals = np.array([r.confidence_interval for r in res.items])  # type: ignore
            return EstimatorArrayResult(values, variances, confidence_intervals)
        return cast(EstimatorResult, res)

    @abstractmethod
    def _preprocessing(
        self, circuit: QuantumCircuit, observable: SparsePauliOp
    ) -> Union[list[QuantumCircuit], tuple[QuantumCircuit, list[QuantumCircuit]]]:
        return NotImplemented

    @abstractmethod
    def _postprocessing(
        self, result: Union[dict, BaseResult, Result]
    ) -> Union[EstimatorResult, EstimatorArrayResult]:
        return NotImplemented
