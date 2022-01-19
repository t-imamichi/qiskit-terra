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
Shot Backend wrapper class
"""

from __future__ import annotations

import logging
from typing import Optional, Union

from qiskit.circuit import QuantumCircuit
from qiskit.providers.backend import BackendV1 as Backend
from qiskit.result import Result
from qiskit.utils.backend_utils import is_aer_provider

from ..backends import BaseBackendWrapper
from ..results import SamplerResult
from .base_sampler import BaseSampler

logger = logging.getLogger(__name__)


class LargeShotSampler(BaseSampler):
    """Sampler class that can deal with a large number of shots"""

    def __init__(
        self,
        backend: Union[Backend, BaseBackendWrapper],
        circuits: Optional[Union[QuantumCircuit, list[QuantumCircuit]]] = None,
    ):
        """ """
        super().__init__(backend=backend, circuits=circuits)

        config = self._backend.backend.configuration()
        self._max_shots = config.max_shots
        if hasattr(config, "max_experiments"):
            self._max_experiments = config.max_experiments
        else:
            self._max_experiments = 1_000_000 if is_aer_provider(self._backend.backend) else 1
            logger.info(
                "no max_experiments for backend '%s'. Set %d as max_experiments.",
                self._backend.backend.name(),
                self._max_experiments,
            )
        self._num_circuits = 0
        self._num_splits = 0
        self._raw_results: list[Result] = []

    @property
    def max_shots(self) -> int:
        """
        The maximum number of shots of the backend

        Returns:
            max_shots
        """
        return self._max_shots

    @property
    def max_experiments(self) -> int:
        """
        The maximum number of circuits in a job for the backend

        Returns:
            max_experiments
        """
        return self._max_experiments

    def _split_experiments(self, shots: int) -> list[tuple[list[QuantumCircuit], int]]:
        """Split circuits so that number of circuits fits in max_experiments"""
        assert self._num_circuits > self._max_experiments
        ret = []
        remaining_shots = shots
        splits = []
        for i in range(0, self._num_circuits, self._max_experiments):
            splits.append(self._circuits[i : min(i + self._max_experiments, self._num_circuits)])
        self._num_splits = len(splits)
        logger.info(
            "Number of circuits %d, Number of splits: %d", self._num_circuits, self._num_splits
        )
        while remaining_shots > 0:
            shots = min(remaining_shots, self._max_shots)
            remaining_shots -= shots
            for circs in splits:
                ret.append((circs, shots))
        return ret

    def _copy_experiments(self, shots: int, exact: bool) -> list[tuple[list[QuantumCircuit], int]]:
        """Copy circuits in the same job to realize the given number of shots"""
        assert self._num_circuits <= self._max_experiments
        max_copies = self._max_experiments // self._num_circuits
        ret = []
        remaining_shots = shots
        while remaining_shots > 0:
            num_copies, rem = divmod(remaining_shots, self._max_shots)
            if rem:
                num_copies += 1
            num_copies = min(num_copies, max_copies)

            shots, rem = divmod(remaining_shots, num_copies)
            if rem and not exact:
                shots += 1
            shots = min(shots, self._max_shots)
            logger.info(
                "Number of circuits %d, number of shots: %d, number of copies: %d, "
                "total number of shots: %d",
                self._num_circuits,
                shots,
                num_copies,
                shots * num_copies,
            )
            remaining_shots -= shots * num_copies
            ret.append((self._circuits * num_copies, shots))
        return ret

    # pylint: disable=arguments-differ
    def run(  # type: ignore
        self,
        append: bool = False,
        exact_shots: bool = True,
        **run_options,
    ) -> SamplerResult:
        """
        TODO

        Returns:
            list of counts
        """
        if "circuits" in run_options:
            self._circuits = run_options["circuits"]
            del run_options["circuits"]
        if "shots" in run_options:
            shots = run_options["shots"]
            del run_options["shots"]
        else:
            shots = self._backend.backend.options.shots
        self._num_circuits = len(self._circuits)
        if self._num_circuits > self._max_experiments:
            circs_shots = self._split_experiments(shots)
        else:
            circs_shots = self._copy_experiments(shots, exact_shots)
        results = []
        for circs, shots in circs_shots:
            result = self._backend.run(circs, shots=shots, **run_options)
            results.append(result)
        if append:
            self._raw_results.extend(results)
        else:
            self._raw_results = results
        counts = self._get_counts(self._raw_results)
        metadata = [res.header.metadata for result in results for res in result.results]
        return SamplerResult(
            counts=counts,
            shots=int(sum(counts[0].values())),
            raw_results=self._raw_results,
            metadata=metadata,
        )
