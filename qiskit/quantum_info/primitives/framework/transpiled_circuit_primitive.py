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
Evaluator class base class
"""
from __future__ import annotations

import sys
from typing import Union

from qiskit import QuantumCircuit
from qiskit.providers import BackendV1 as Backend

from ..backends import BaseBackendWrapper, ShotBackendWrapper, ShotResult
from ..results.base_result import BaseResult
from .base_primitive import BasePrimitive

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol  # pylint: disable=import-error


class Postprocessing(Protocol):
    """Postprocessing Callback Protocol (PEP544)"""

    def __call__(self, result: Union[ShotResult, dict]) -> BaseResult:
        ...


class TranspiledCircuitPrimitive(BasePrimitive):
    """
    Evaluator for transpiled circuits.
    """

    def __init__(
        self,
        transpiled_circuits: list[QuantumCircuit],
        postprocessing: Postprocessing,
        backend: Union[Backend, BaseBackendWrapper, ShotBackendWrapper],
    ):
        """
        Args:
            backend: backend
        """
        super().__init__(backend=backend)
        self._injected_postprocessing = postprocessing
        self._transpiled_circuits = transpiled_circuits

    def _postprocessing(self, result) -> BaseResult:
        return self._injected_postprocessing(result)
