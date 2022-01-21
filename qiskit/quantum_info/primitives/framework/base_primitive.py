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

import copy
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, cast

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.providers import BackendV1 as Backend
from qiskit.providers import Options
from qiskit.result import Result

from ..backends import (
    BackendWrapper,
    BaseBackendWrapper,
)
from ..results import CompositeResult
from ..results.base_result import BaseResult

PreprocessedCircuits = Union[
    "list[QuantumCircuit]", "list[tuple[QuantumCircuit, list[QuantumCircuit]]]"
]


class BasePrimitive(ABC):
    """
    Base class for primitives.
    """

    def __init__(
        self,
        backend: Union[Backend, BaseBackendWrapper],
        transpile_options: Optional[dict] = None,
    ):
        """
        Args:
            backend: backend
        """
        self._backend: BaseBackendWrapper
        self._backend = BackendWrapper.from_backend(backend)
        self._run_options = Options()
        self._is_closed = False

        self._transpile_options = Options()
        if transpile_options is not None:
            self.set_transpile_options(**transpile_options)

        self._preprocessed_circuits: Optional[PreprocessedCircuits] = None
        self._transpiled_circuits: Optional[list[QuantumCircuit]] = None

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, trace):
        self._is_closed = True

    def __call__(
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
        return self.run(parameters, **run_options)

    @property
    def run_options(self) -> Options:
        """Return options values for the evaluator.
        Returns:
            run_options
        """
        return self._run_options

    def set_run_options(self, **fields) -> BasePrimitive:
        """Set options values for the evaluator.

        Args:
            fields: The fields to update the options
        Returns:
            self
        """
        self._run_options.update_options(**fields)
        return self

    @property
    def transpile_options(self) -> Options:
        """Return the transpiler options for transpiling the circuits."""
        return self._transpile_options

    def set_transpile_options(self, **fields) -> BasePrimitive:
        """Set the transpiler options for transpiler.
        Args:
            fields: The fields to update the options.
        Returns:
            self.
        Raises:
            QiskitError: if the instance has been closed.
        """
        if self._is_closed:
            raise QiskitError("The primitive has been closed.")

        self._transpile_options.update_options(**fields)
        return self

    @property
    def backend(self) -> Backend:
        """Backend

        Returns:
            backend
        """
        return self._backend.backend

    @property
    def preprocessed_circuits(self) -> Optional[PreprocessedCircuits]:
        """
        Preprocessed quantum circuits produced by preprocessing

        Returns:
            List of the transpiled quantum circuit
        Raises:
            QiskitError: if the instance has been closed.
        """
        if self._is_closed:
            raise QiskitError("The primitive has been closed.")
        return self._preprocessed_circuits

    @property
    def transpiled_circuits(self) -> list[QuantumCircuit]:
        """
        Transpiled quantum circuits.

        Returns:
            List of the transpiled quantum circuit
        Raises:
            QiskitError: if the instance has been closed.
        """
        if self._is_closed:
            raise QiskitError("The primitive has been closed.")
        if self._transpiled_circuits is None:
            self._transpile()
        return self._transpiled_circuits

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
        """
        Returns:
            The running result.
        Raises:
            QiskitError: if the instance has been closed.
            TypeError: if the shape of parameters is invalid.
        """
        if self._is_closed:
            raise QiskitError("The primitive has been closed.")

        # Bind parameters
        # TODO: support Aer parameter bind after https://github.com/Qiskit/qiskit-aer/pull/1317
        if parameters is None:
            bound_circuits = self.transpiled_circuits
        else:
            parameters = np.asarray(parameters, dtype=np.float64)
            if parameters.ndim == 1:
                bound_circuits = [
                    circ.bind_parameters(parameters)  # type: ignore
                    for circ in self.transpiled_circuits
                ]
            elif parameters.ndim == 2:
                bound_circuits = [
                    circ.bind_parameters(parameter)
                    for parameter in parameters
                    for circ in self.transpiled_circuits
                ]
            else:
                raise TypeError("The number of array dimension must be 1 or 2.")

        # Run
        run_opts = copy.copy(self.run_options)
        run_opts.update_options(**run_options)

        results = self._backend.run(circuits=bound_circuits, **run_opts.__dict__)

        if parameters is None or isinstance(parameters, np.ndarray) and parameters.ndim == 1:
            ret_result = self._postprocessing(results)
        else:
            if isinstance(results, Result):
                postprocessed = [
                    self._postprocessing(
                        results.results[
                            i
                            * len(self.transpiled_circuits) : (i + 1)
                            * len(self.transpiled_circuits)
                        ]
                    )
                    for i in range(len(parameters))
                ]
            else:
                postprocessed = [
                    self._postprocessing(
                        results[
                            i
                            * len(self.transpiled_circuits) : (i + 1)
                            * len(self.transpiled_circuits)
                        ]
                    )
                    for i in range(len(parameters))
                ]
            ret_result = CompositeResult(postprocessed)

        return ret_result

    def _transpile(self):
        self._transpiled_circuits = cast(
            "list[QuantumCircuit]",
            transpile(self.preprocessed_circuits, self.backend, **self.transpile_options.__dict__),
        )

    @abstractmethod
    def _postprocessing(self, result: Union[Result, BaseResult, dict]) -> BaseResult:
        return NotImplemented
