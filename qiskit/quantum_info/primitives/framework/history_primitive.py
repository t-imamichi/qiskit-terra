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
Evaluator with history
"""
from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from qiskit.result import Result

from ..results.base_result import BaseResult
from .base_primitive import BasePrimitive, PreprocessedCircuits


class HistoryPrimitive(BasePrimitive):
    """Primitive with history"""

    def __init__(self, primitive: BasePrimitive):
        """
        Args:
            primitive:
        """
        super().__init__(
            backend=primitive._backend, transpile_options=primitive.transpile_options.__dict__
        )
        self._primitive = primitive
        self._history: list[BaseResult] = []

    @property
    def history(self) -> list[BaseResult]:
        """History of evaluation.

        Return:
            history
        """
        return self._history

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
    ) -> BaseResult:

        result = super().run(parameters, **run_options)
        self._history.append(result)
        return result

    @property
    def preprocessed_circuits(self) -> PreprocessedCircuits:
        return self._primitive.preprocessed_circuits

    @property
    def transpiled_circuits(self):
        return self._primitive.transpiled_circuits

    def _postprocessing(self, result: Union[Result, BaseResult, dict]) -> BaseResult:
        return self._primitive._postprocessing(result)
