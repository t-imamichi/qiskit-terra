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
Expectation value gradient class with Opflow
"""

from __future__ import annotations

from typing import Any, Optional, Union, cast

import numpy as np

from qiskit.circuit import ParameterExpression
from qiskit.exceptions import QiskitError
from qiskit.opflow import CircuitStateFn, ListOp, SummedOp
from qiskit.opflow.gradients import Gradient
from qiskit.quantum_info import Pauli

from ..estimator.base_estimator import BaseEstimator
from ..results import CompositeResult, EstimatorGradientResult, EstimatorResult
from .estimator_gradient import BaseEstimatorGradient

Pauli_Z = Pauli("Z")


class OpflowGradient(BaseEstimatorGradient):
    """TODO"""

    def __init__(self, estimator: BaseEstimator, grad_method: str = "param_shift"):
        super().__init__(estimator)
        self._grad_method = grad_method
        self._grad: list[list[tuple[Union[float, ParameterExpression], BaseEstimator]]] = []

    def _preprocessing(self):
        if self._grad:
            return

        expval = self._estimator
        op = CircuitStateFn(expval.circuit)
        grad = cast(ListOp, Gradient(self._grad_method).convert(op))
        assert len(grad.oplist) == len(expval.circuit.parameters)

        def extract_circuits(list_op: ListOp):
            lst = []
            if self._grad_method == "lin_comb":
                mul = 2
                for i, op in enumerate(list_op.oplist):
                    op = cast(CircuitStateFn, op)
                    observable = expval.observable.expand(Pauli_Z)
                    new_expval = expval.__class__(
                        circuit=op.primitive, observable=observable, backend=expval.backend
                    )
                    lst.append((mul * list_op.coeff * op.coeff ** 2, new_expval))  # type: ignore
            elif self._grad_method in ["param_shift", "fin_diff"]:
                if "shift_constant" in list_op.combo_fn.keywords:  # type: ignore
                    mul = list_op.combo_fn.keywords["shift_constant"]  # type: ignore
                else:
                    mul = 1
                for i, op in enumerate(list_op.oplist):
                    op = cast(CircuitStateFn, op)
                    observable = (-1) ** i * expval.observable
                    new_expval = expval.__class__(
                        circuit=op.primitive, observable=observable, backend=expval.backend
                    )
                    lst.append((mul * list_op.coeff * op.coeff ** 2, new_expval))  # type: ignore
            else:
                raise QiskitError(
                    f"internal error: unsupported gradient method {self._grad_method}"
                )

            return lst

        for op in grad.oplist:
            if isinstance(op, SummedOp):
                lst = []
                for op2 in op.oplist:
                    for coeff, new_expval in extract_circuits(cast(ListOp, op2)):
                        lst.append((op.coeff * coeff, new_expval))
                self._grad.append(lst)
            elif isinstance(op, ListOp):
                self._grad.append(extract_circuits(op))
            else:
                raise QiskitError("internal error: `op` should be ListOp or SummedOp.")

    def _eval_parameters(
        self, parameters: np.ndarray[Any, np.dtype[np.float64]]
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        pass

    def _compute_gradient(self, results: CompositeResult, shape) -> EstimatorGradientResult:
        pass

    def run(  # pylint: disable=signature-differs
        self,
        parameters: Optional[
            Union[list[float], list[list[float]], np.ndarray[Any, np.dtype[np.float64]]]
        ] = None,
        **run_options,
    ) -> EstimatorGradientResult:
        if parameters is None:
            raise TypeError("parameters is None")

        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.ndim not in [1, 2]:
            raise ValueError("parameters should be a 1D vector or 2D vectors")

        num_param_sets = 1 if parameters.ndim == 1 else parameters.shape[0]

        param_map = {}
        for j, param in enumerate(self._estimator.circuit.parameters):
            if parameters.ndim == 1:
                param_map[param, 0] = parameters[j]
            else:
                for i in range(num_param_sets):
                    param_map[param, i] = parameters[i, j]

        self._preprocessing()

        ret = np.zeros(parameters.shape)
        for j, lst in enumerate(self._grad):
            for coeff, expval in lst:
                bound_coeff: Union[float, np.ndarray]
                if isinstance(coeff, ParameterExpression):
                    bound_coeff = np.zeros(num_param_sets)
                    for i in range(num_param_sets):
                        local_map = {param: param_map[param, i] for param in coeff.parameters}
                        bound_coeff[i] = coeff.bind(local_map)
                else:
                    bound_coeff = coeff
                result = expval.run(parameters, **run_options)
                if isinstance(result, EstimatorResult):
                    ret[j] += bound_coeff * result.value
                else:
                    ret[:, j] += bound_coeff * result.values

        return EstimatorGradientResult(values=ret)
