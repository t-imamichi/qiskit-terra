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
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Optional, Union, cast

from qiskit.circuit import QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.providers import BackendV1 as Backend
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.primitives.framework.base_primitive import (
    BasePrimitive,
    PreprocessedCircuits,
)
from qiskit.result import Result

from ..backends import BaseBackendWrapper
from ..results import CompositeResult, EstimatorArrayResult
from ..results.base_result import BaseResult
from ..sampler import BaseSampler
from .utils import init_circuit, init_observable

if TYPE_CHECKING:
    from typing import Any

    import numpy as np


@dataclass(frozen=True)
class Group:
    """The dataclass represents indices of circuit and observable."""

    circuit_index: int
    observable_index: int


class BaseEstimator(BasePrimitive, ABC):
    """
    Expectation Value class
    """

    def __init__(
        self,
        circuits: list[Union[QuantumCircuit, Statevector]],
        observables: list[Union[BaseOperator, PauliSumOp]],
        sampler: Union[Backend, BaseBackendWrapper, BaseSampler],
    ):
        """ """
        if not isinstance(circuits, list):
            raise TypeError("circuits must be list.")
        if not isinstance(observables, list):
            raise TypeError("observables must be list.")

        super().__init__(backend=sampler.backend if isinstance(sampler, BaseSampler) else sampler)
        self._sampler = sampler
        self._circuits = [init_circuit(circuit) for circuit in circuits]
        self._observables = [init_observable(observable) for observable in observables]
        self._grouping = [
            Group(i, j) for i in range(len(circuits)) for j in range(len(observables))
        ]
        self._transpiled_circuits_cache: dict[list[Group], list[QuantumCircuit]] = {}

    @property
    def circuits(self) -> list[QuantumCircuit]:
        """Quantum Circuits that represents quantum states.

        Returns:
            quantum states
        """
        return self._circuits

    @property
    def observables(self) -> list[SparsePauliOp]:
        """
        SparsePauliOp that represents observable

        Returns:
            observable
        """
        return self._observables

    def set_transpile_options(self, **fields) -> BaseEstimator:
        """Set the transpiler options for transpiler.

        Args:
            fields: The fields to update the options
        Returns:
            self
        """
        self._transpiled_circuits = None
        self._transpiled_circuits_cache = {}
        super().set_transpile_options(**fields)
        return self

    @property
    def preprocessed_circuits(
        self,
    ) -> PreprocessedCircuits:
        """
        Transpiled quantum circuits produced by preprocessing

        Returns:
            List of the transpiled quantum circuit
        """
        if self._preprocessed_circuits is None:
            self._preprocessed_circuits = self._preprocessing(self.circuits, self.observables)
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
    ) -> EstimatorArrayResult:
        if "grouping" in run_options:
            self._grouping = [
                g if isinstance(g, Group) else Group(g[0], g[1]) for g in run_options["grouping"]
            ]
            del run_options["grouping"]
        else:
            self._grouping = [
                Group(i, j) for i in range(len(self.circuits)) for j in range(len(self.observables))
            ]
        result = super().run(parameters, **run_options)
        if isinstance(result, CompositeResult):
            return cast(
                EstimatorArrayResult,
                reduce(lambda a, b: a + b, result.items),  # type: ignore # pylint: disable=no-member
            )
        return cast(EstimatorArrayResult, result)

    @abstractmethod
    def _preprocessing(
        self, circuits: list[QuantumCircuit], observables: list[SparsePauliOp]
    ) -> Union[list[QuantumCircuit], list[tuple[QuantumCircuit, list[QuantumCircuit]]]]:
        return NotImplemented

    @abstractmethod
    def _postprocessing(self, result: Union[dict, BaseResult, Result]) -> EstimatorArrayResult:
        return NotImplemented

    def _transpile(self):

        if tuple(self._grouping) in self._transpiled_circuits_cache:
            self._transpiled_circuits = self._transpiled_circuits_cache[tuple(self._grouping)]
        else:
            super()._transpile()
            self._transpiled_circuits_cache[tuple(self._grouping)] = self._transpiled_circuits
