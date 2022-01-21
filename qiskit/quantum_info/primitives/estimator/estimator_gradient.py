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
Expectation value gradient class
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union, cast

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.result import Result

from ..framework.base_primitive import BasePrimitive, PreprocessedCircuits
from ..results import (
    CompositeResult,
    EstimatorArrayResult,
    EstimatorGradientResult,
)
from ..results.base_result import BaseResult
from .base_estimator import BaseEstimator


class BaseEstimatorGradient(BasePrimitive, ABC):
    """
    Base class for expectation value gradient
    """

    def __init__(
        self,
        estimator: BaseEstimator,
    ):
        self._estimator = estimator
        super().__init__(
            backend=self._estimator._backend,
            transpile_options=self._estimator.transpile_options.__dict__,
        )

    @property
    def preprocessed_circuits(self) -> PreprocessedCircuits:
        """
        Preprocessed quantum circuits produced by preprocessing

        Returns:
            List of the transpiled quantum circuit
        """
        return self._estimator.preprocessed_circuits

    @property
    def transpiled_circuits(self) -> list[QuantumCircuit]:
        """
        Transpiled quantum circuits.

        Returns:
            List of the transpiled quantum circuit
        """
        return self._estimator.transpiled_circuits

    @abstractmethod
    def _eval_parameters(
        self, parameters: "np.ndarray[Any, np.dtype[np.float64]]"
    ) -> "np.ndarray[Any, np.dtype[np.float64]]":
        return NotImplemented

    @abstractmethod
    def _compute_gradient(self, results: CompositeResult, shape) -> EstimatorGradientResult:
        return NotImplemented

    def run(
        self,
        parameters: Optional[
            Union[list[float], list[list[float]], np.ndarray[Any, np.dtype[np.float64]]]
        ] = None,
        **run_options,
    ) -> EstimatorGradientResult:
        """TODO"""
        if parameters is None:
            raise ValueError()

        parameters = np.asarray(parameters, dtype=np.float64)
        if len(parameters.shape) not in [1, 2]:
            raise ValueError("parameters should be a 1D vector or 2D vectors")
        param_array = self._eval_parameters(parameters)
        results = cast(CompositeResult, super().run(param_array, **run_options))
        return self._compute_gradient(results, parameters.shape)

    def _postprocessing(self, result: Union[Result, BaseResult, dict]) -> EstimatorArrayResult:
        return self._estimator._postprocessing(result)


class FiniteDiffGradient(BaseEstimatorGradient):
    """
    Finite difference of expectation values
    """

    def __init__(self, estimator: BaseEstimator, epsilon: float):
        super().__init__(estimator)
        self._epsilon = epsilon

    def _eval_parameters(
        self, parameters: "np.ndarray[Any, np.dtype[np.float64]]"
    ) -> "np.ndarray[Any, np.dtype[np.float64]]":
        if len(parameters.shape) == 1:
            parameters = parameters.reshape((1, parameters.shape[0]))
        dim = parameters.shape[-1]
        ret = []
        for param in parameters:
            ret.append(param)
            for i in range(dim):
                ei = param.copy()
                ei[i] += self._epsilon
                ret.append(ei)
        return np.array(ret)

    def _compute_gradient(self, results: CompositeResult, shape) -> EstimatorGradientResult:
        values = np.array([r.values[0] for r in results.items])  # type: ignore
        dim = shape[-1]
        array = values.reshape((values.shape[0] // (dim + 1), dim + 1))
        ret = []
        for values in array:
            grad = np.zeros(dim)
            f_ref = values[0]
            for i, f_i in enumerate(values[1:]):
                grad[i] = (f_i - f_ref) / self._epsilon
            ret.append(grad)
        grad = np.array(ret).reshape(shape)
        return EstimatorGradientResult(values=grad)


class ParameterShiftGradient(BaseEstimatorGradient):
    """
    Gradient of expectation values by parameter shift
    """

    def __init__(self, estimator: BaseEstimator):
        super().__init__(estimator)
        self._epsilon = np.pi / 2

    def _eval_parameters(
        self, parameters: "np.ndarray[Any, np.dtype[np.float64]]"
    ) -> "np.ndarray[Any, np.dtype[np.float64]]":
        if len(parameters.shape) == 1:
            parameters = parameters.reshape((1, parameters.shape[0]))
        dim = parameters.shape[-1]
        ret = []
        for param in parameters:
            for i in range(dim):
                ei = param.copy()
                ei[i] += self._epsilon
                ret.append(ei)

                ei = param.copy()
                ei[i] -= self._epsilon
                ret.append(ei)

        return np.array(ret)

    def _compute_gradient(self, results: CompositeResult, shape) -> EstimatorGradientResult:
        values = np.array([r.values[0] for r in results.items])  # type: ignore
        dim = shape[-1]
        array = values.reshape((values.shape[0] // (2 * dim), 2 * dim))
        div = 2 * np.sin(self._epsilon)
        ret = []
        for values in array:
            grad = np.zeros(dim)
            for i in range(dim):
                f_plus = values[2 * i]
                f_minus = values[2 * i + 1]
                grad[i] = (f_plus - f_minus) / div
            ret.append(grad)
        grad = np.array(ret).reshape(shape)
        return EstimatorGradientResult(values=grad)
