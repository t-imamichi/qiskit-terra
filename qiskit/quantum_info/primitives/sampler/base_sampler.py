# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
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

from abc import ABC
from typing import Any, Optional, Union, cast

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.providers.backend import BackendV1 as Backend
from qiskit.quantum_info.primitives.framework import BasePrimitive
from qiskit.result import Result

from ..backends import BaseBackendWrapper
from ..results import SamplerResult


class BaseSampler(BasePrimitive, ABC):
    """
    Base Sampler class
    """

    def __init__(
        self,
        backend: Union[Backend, BaseBackendWrapper],
        circuits: Optional[Union[QuantumCircuit, list[QuantumCircuit]]] = None,
    ):
        """ """
        super().__init__(backend=backend)
        if circuits is None:
            self._circuits = None
        elif isinstance(circuits, list):
            self._circuits = circuits
        else:
            self._circuits = [circuits]

    @classmethod
    def from_backend(
        cls, backend: Union[Backend, BaseBackendWrapper, BaseSampler]
    ) -> "BaseSampler":
        """Sampler based on a backend"""
        if not isinstance(backend, BaseSampler):
            return cls(backend=backend)
        return backend

    @property
    def circuits(self) -> list[QuantumCircuit]:
        """Quantum Circuit that represents quantum state.

        Returns:
            quantum state
        """
        return self._circuits

    @circuits.setter
    def circuits(self, circuits: Union[QuantumCircuit, list[QuantumCircuit]]):
        self._circuits = circuits if isinstance(circuits, list) else [circuits]

    @property
    def preprocessed_circuits(self) -> Optional[list[QuantumCircuit]]:
        return self._circuits

    def run(
        self,
        parameters: Optional[
            Union[
                list[float],
                list[list[float]],
                np.ndarray[Any, np.dtype[np.float64]],
            ]
        ] = None,
        **run_options,
    ) -> SamplerResult:
        if "circuits" in run_options:
            self._circuits = run_options["circuits"]
            del run_options["circuits"]
        return cast(SamplerResult, super().run(parameters, **run_options))

    @property
    def backend(self) -> Backend:
        """
        TODO

        Returns:
            backend
        """
        return self._backend.backend

    def _postprocessing(self, result: Result) -> SamplerResult:
        """TODO"""
        raw_results = [result]
        counts = self._get_counts(raw_results)
        metadata = [res.header.metadata for result in raw_results for res in result.results]
        return SamplerResult(
            counts=counts,
            shots=int(sum(counts[0].values())),
            raw_results=raw_results,
            metadata=metadata,
        )
