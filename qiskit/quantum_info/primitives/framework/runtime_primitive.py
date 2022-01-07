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
Expectation value serialization for Qiskit Runtime
"""

from __future__ import annotations

import copy
import json
from typing import Any, Optional, Union

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.utils.backend_utils import has_aer, has_ibmq

from ..backends import ShotResult
from ..estimator import BaseEstimator, PauliEstimator
from ..results import EstimatorArrayResult, EstimatorResult
from ..results.base_result import BaseResult
from .base_primitive import BasePrimitive, PreprocessedCircuits

if has_ibmq():
    from qiskit.providers.ibmq.runtime import (  # pylint: disable=no-name-in-module, import-error
        RuntimeDecoder,
        RuntimeEncoder,
    )

if has_aer():
    from qiskit import Aer


class RuntimePrimitive(BasePrimitive):
    """Evaluator on runtime"""

    def __init__(self, primitive: BasePrimitive, provider):
        """
        Args:
            evaluator:
        """
        if not has_ibmq():
            pass  # Raise MissingOptionalLibraryError
        super().__init__(
            backend=primitive._backend, transpile_options=primitive.transpile_options.__dict__
        )
        self._evaluator = primitive
        self._provider = provider

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
        run_opts = copy.copy(self._evaluator.run_options)
        run_opts.update_options(**run_options)

        options = {"backend_name": self._evaluator.backend.name()}
        if isinstance(self._evaluator, BaseEstimator):
            runtime_inputs = {
                "state": self._evaluator.circuit,
                "observable": self._evaluator.observable,
                "parameters": parameters,
                "class_name": self._evaluator.__class__.__name__,
                "transpile_options": self._evaluator.transpile_options.__dict__,
                "run_options": run_opts.__dict__,
            }
        else:
            raise QiskitError(f"{self._evaluator.__class__.__name__} is not supported for runtime.")

        job = self._provider.runtime.run(
            program_id="estimator",
            options=options,
            inputs=runtime_inputs,
            image="expval:latest",
        )
        job_result = job.result()

        result: BaseResult
        if "value" in job_result:
            result = EstimatorResult(**job_result)
        elif "values" in job_result:
            result = EstimatorArrayResult(**job_result)
        else:
            raise QiskitError("unknown result type")
        return result

    @property
    def preprocessed_circuits(self) -> PreprocessedCircuits:
        return self._evaluator.preprocessed_circuits

    def _postprocessing(self, result: Union[ShotResult, dict]) -> BaseResult:
        return self._evaluator._postprocessing(result)


def runtime_dump(
    expval: BaseEstimator, parameters: Union[list[float], list[list[float]], np.ndarray]
):
    """TODO"""
    ret = {
        "state": expval.circuit,
        "observable": expval.observable,
        "parameters": parameters,
        "class_name": expval.__class__.__name__,
        "transpile_options": expval.transpile_options.__dict__,
        "run_options": expval.run_options.__dict__,
    }
    return json.dumps(ret, cls=RuntimeEncoder, indent=2)


def runtime_load(ser: str) -> tuple[BaseEstimator, Union[list[float], list[list[float]]]]:
    """TODO"""
    params = json.loads(ser, cls=RuntimeDecoder)
    if params["class_name"] == "PauliEstimator":
        cls = PauliEstimator
    else:
        raise QiskitError(f"Unexpected expectation value class name {params['class_name']}")
    state = params["state"]
    observable = params["observable"]
    backend = Aer.get_backend("aer_simulator")
    parameters = params["parameters"]
    return cls(state, observable, backend), parameters
