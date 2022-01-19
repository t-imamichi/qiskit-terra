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
Joint primitive class
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from qiskit.result import Result

from ...framework.base_primitive import BasePrimitive
from ...results import CompositeResult
from ...results.base_result import BaseResult

if TYPE_CHECKING:
    from typing import Any


class JointPrimitive(BasePrimitive):
    """Joint Primitive"""

    def __init__(self, primitives: list[BasePrimitive]):
        """hoge"""

        self._primitives = primitives

        self._counter = 0
        super().__init__(primitives[0]._backend)

        for primitive in primitives:
            if primitive.backend != self.backend:
                raise ValueError("")
            # Should we update the run_options?
            # self.run_options.update_options(**primitive.run_options.__dict__)

    @property
    def transpiled_circuits(self):
        if self._transpiled_circuits is None:
            self._transpiled_circuits = sum(
                [primitive.transpiled_circuits for primitive in self._primitives], []
            )
        return self._transpiled_circuits

    def _postprocessing(self, result: Result) -> BaseResult:
        current_counter = self._counter
        self._counter += 1
        if self._counter == len(self._primitives):
            self._counter = 0
        return self._primitives[current_counter]._postprocessing(result)

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
    ) -> CompositeResult:
        run_opts = copy.copy(self.run_options)
        run_opts.update_options(**run_options)
        run_opts_dict = run_opts.__dict__

        if len(parameters) != len(self._primitives):
            raise TypeError("Length is different.")

        if parameters is None:
            circuits = self.transpiled_circuits
        else:
            parameters = np.asarray(parameters, dtype=np.float64)
            if parameters.ndim == 2:
                circuits = [
                    circ.bind_parameters(param)
                    for param in parameters
                    for primitive in self._primitives
                    for circ in primitive.transpiled_circuits
                ]
            elif parameters.ndim == 1:
                circuits = [
                    circ.bind_parameters(parameters)  # type: ignore
                    for primitive in self._primitives
                    for circ in primitive.transpiled_circuits
                ]

        results = self._backend.run(circuits, **run_opts_dict)

        accum = 0
        postprocessed = []
        for primitive in self._primitives:
            postprocessed.append(
                self._postprocessing(
                    results[accum : accum + len(primitive.preprocessed_circuits)]  # type:ignore
                )
            )

        return CompositeResult(postprocessed)
