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
On-chip parallelization sampler
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Optional, Union, cast

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.providers.backend import BackendV1 as Backend
from qiskit.result import Counts, Result

from ..backends import BaseBackendWrapper
from ..results import SamplerResult
from ..results.base_result import BaseResult
from .base_sampler import BaseSampler

logger = logging.getLogger(__name__)


class OnchipParallelSampler(BaseSampler):
    """On-chip parallelization sampler"""

    def __init__(
        self,
        backend: Union[Backend, BaseBackendWrapper],
        layout: Union[list[int], list[list[int]]],
        circuits: Optional[Union[QuantumCircuit, list[QuantumCircuit]]] = None,
    ):
        """ """
        super().__init__(backend=backend, circuits=circuits)
        self._layout: list[list[int]] = [layout] if isinstance(layout[0], int) else layout
        if len(set(len(e) for e in self._layout)) != 1:
            logger.fatal(
                "Qubit layout is not consistent. " "All layouts should have the same size: %s",
                self._layout,
            )
        self._validate_circuits()
        config = self._backend.backend.configuration()
        self._num_qubits = config.num_qubits

    @property
    def layout(self) -> list[list[int]]:
        """Return layout"""
        return self._layout

    def _validate_circuits(self) -> bool:
        if self._circuits is None:
            return True
        nums_qubits = [circ.num_qubits for circ in self._circuits]
        if len(set(nums_qubits)) != 1:
            logger.fatal(
                "Numbers of qubits of circuits are not consistent. "
                "All circuits should have the same number of qubits: %d",
                nums_qubits,
            )
            return False
        return True

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
            self._validate_circuits()
            del run_options["circuits"]
        if "layout" in run_options:
            layout = run_options["layout"]
            self._layout = [layout] if isinstance(layout[0], int) else layout
            del run_options["layout"]
        self._circuits = self._embed_circuits()
        self._validate_circuits()
        if parameters is None:
            new_parameters = parameters
        else:
            new_parameters = parameters * len(self._layout)
        initial_layout = sum(self._layout, start=[])
        self.set_transpile_options(initial_layout=initial_layout)
        return cast(SamplerResult, super().run(new_parameters, **run_options))

    def _embed_circuits(self):
        if self._circuits[0].num_qubits != len(self._layout[0]):
            logger.fatal(
                "The number of qubits (%d) does not match with "
                "the number of qubits of the qubit layout (%d)",
                self._circuits[0].num_qubits,
                len(self._layout[0]),
            )
        total_num_qubits = self._circuits[0].num_qubits * len(self._layout)
        if total_num_qubits > self._num_qubits:
            logger.fatal(
                "Total number of qubits (%d) exceeds the number of qubits (%d) of the backend",
                total_num_qubits,
                self._num_qubits,
            )
        circuits = []
        for circ in self._circuits:
            new_circ = circ
            for _ in range(len(self._layout) - 1):
                new_circ = circ.tensor(new_circ)
            new_circ.metadata = circ.metadata
            circuits.append(new_circ)

        return circuits

    def _postprocessing(self, result: Union[Result, BaseResult, dict]) -> SamplerResult:
        """TODO"""
        if not isinstance(result, Result):
            raise TypeError("result must be an instance of Result.")

        raw_results = [result]
        raw_counts = self._get_counts(raw_results)  # type: ignore
        new_counts = []
        step = len(self._layout[0])
        for counts in raw_counts:
            counter = Counter()
            for key, value in counts.items():
                for i in range(0, len(key), step):
                    counter[key[i : i + step]] += value
            new_counts.append(Counts(counter))

        metadata = [
            res.header.metadata
            for result in raw_results
            for res in result.results  # type:ignore # pylint: disable=no-member
        ]

        return SamplerResult(
            counts=new_counts,
            shots=int(sum(new_counts[0].values())),
            raw_results=raw_results,
            metadata=metadata,
        )
