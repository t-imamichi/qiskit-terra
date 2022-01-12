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
from collections import Counter
from typing import Optional, Union

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import BackendV1 as Backend
from qiskit.quantum_info.primitives.framework import BasePrimitive
from qiskit.result import Counts, Result

from ..backends import BaseBackendWrapper, ReadoutErrorMitigation
from ..results import SamplerResult
from ..results.base_result import BaseResult


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
        cls, backend: Union[Backend, BaseBackendWrapper, "BaseSampler"]
    ) -> "BaseSampler":
        """Sampler based on a backend"""
        if isinstance(backend, (Backend, BaseBackendWrapper)):
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

    # pylint: disable=arguments-differ
    def run(
        self,
        **run_options,
    ) -> Union[SamplerResult]:
        if "circuits" in run_options:
            self._circuits = run_options["circuits"]
            del run_options["circuits"]
        if "shots" in run_options:
            shots = run_options["shots"]
            del run_options["shots"]
        else:
            shots = self._backend.backend.options.shots
        raw_results = [self._backend.run(self._circuits, shots=shots, **run_options)]
        counts = self._get_counts(raw_results)
        metadata = [res.header.metadata for result in raw_results for res in result.results]
        return SamplerResult(
            counts=counts,
            shots=int(sum(counts[0].values())),
            raw_results=raw_results,
            metadata=metadata,
        )

    @property
    def backend(self) -> Backend:
        """
        TODO

        Returns:
            backend
        """
        return self._backend.backend

    def _get_counts(self, results: list[Result]) -> list[Counts]:
        """
        Convert Result to Counts

        Returns:
            list of counts
        Raises:
            QiskitError: if inputs are empty
        """
        if len(results) == 0:
            raise QiskitError("Empty result")
        if isinstance(self._backend, ReadoutErrorMitigation):
            list_counts = self._backend.apply_mitigation(results)
        else:
            list_counts = [result.get_counts() for result in results]
        num_circuits = len(self._circuits)
        counters: list[Counter] = [Counter() for _ in range(num_circuits)]
        i = 0
        for counts in list_counts:
            if isinstance(counts, Counts):
                counts = [counts]
            for count in counts:
                counters[i % num_circuits].update(count)
                i += 1
        # TODO: recover the metadata of Counts
        return [Counts(c) for c in counters]

    def _postprocessing(self, result: Union[SamplerResult, dict]) -> BaseResult:
        """TODO"""
        pass
