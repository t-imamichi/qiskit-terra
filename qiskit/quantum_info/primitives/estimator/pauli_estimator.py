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
Expectation value class
"""

from __future__ import annotations

import copy
import logging
from functools import reduce
from itertools import accumulate
from typing import Any, Optional, Union, cast

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.opflow import AbelianGrouper, PauliSumOp
from qiskit.providers import BackendV1 as Backend
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result import Counts, Result

from ..results import EstimatorArrayResult, SamplerResult
from ..results.base_result import BaseResult
from ..sampler import BaseSampler
from .base_estimator import BaseEstimator, Group

logger = logging.getLogger(__name__)


class PauliEstimator(BaseEstimator):
    """
    Evaluates expectation value using pauli rotation gates.
    """

    def __init__(
        self,
        circuits: list[Union[QuantumCircuit, Statevector]],
        observables: list[Union[BaseOperator, PauliSumOp]],
        sampler: Union[Backend, BaseSampler],
        strategy: bool = True,  # To be str like TPB
    ):
        super().__init__(
            circuits=circuits,
            observables=observables,
            sampler=sampler,
        )
        self._measurement_strategy = strategy

    @property
    def preprocessed_circuits(
        self,
    ) -> list[tuple[QuantumCircuit, list[QuantumCircuit]]]:
        """
        Transpiled quantum circuits produced by preprocessing

        Returns:
            List of the transpiled quantum circuit
        """
        return cast(
            "list[tuple[QuantumCircuit, list[QuantumCircuit]]]", super().preprocessed_circuits
        )

    def _transpile(self):
        """Split Transpile"""
        self._transpiled_circuits = []
        for common_circuit, diff_circuits in self.preprocessed_circuits:
            # 1. transpile a common circuit
            common_circuit = common_circuit.copy()
            num_qubits = common_circuit.num_qubits
            common_circuit.measure_all()
            common_circuit = cast(
                QuantumCircuit,
                transpile(common_circuit, **self.transpile_options.__dict__),
            )
            bit_map = {bit: index for index, bit in enumerate(common_circuit.qubits)}
            layout = [bit_map[qr[0]] for _, qr, _ in common_circuit[-num_qubits:]]
            common_circuit.remove_final_measurements()
            # 2. transpile diff circuits
            transpile_opts = copy.copy(self.transpile_options)
            transpile_opts.update_options(initial_layout=layout)
            diff_circuits = cast(
                "list[QuantumCircuit]",
                transpile(diff_circuits, **transpile_opts.__dict__),
            )
            # 3. combine
            transpiled_circuits = []
            for diff_circuit in diff_circuits:
                transpiled_circuit = common_circuit.copy()
                for creg in diff_circuit.cregs:
                    if creg not in transpiled_circuit.cregs:
                        transpiled_circuit.add_register(creg)
                transpiled_circuit.compose(diff_circuit, inplace=True)
                transpiled_circuit.metadata = diff_circuit.metadata
                transpiled_circuits.append(transpiled_circuit)
            self._transpiled_circuits += transpiled_circuits

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
    ) -> EstimatorArrayResult:
        """
        Returns:
            The running result.
        Raises:
            QiskitError: if the instance has been closed.
            TypeError: if the shape of parameters is invalid.
        """
        if self._is_closed:
            raise QiskitError("The primitive has been closed.")

        if "grouping" in run_options:
            self._grouping = [
                g if isinstance(g, Group) else Group(g[0], g[1]) for g in run_options["grouping"]
            ]
            del run_options["grouping"]
        else:
            self._grouping = [
                Group(i, j) for i in range(len(self.circuits)) for j in range(len(self.observables))
            ]

        # Bind parameters
        # TODO: support Aer parameter bind after https://github.com/Qiskit/qiskit-aer/pull/1317
        # Is it possible to bind in Sampler?
        if parameters is None:
            bound_circuits = self.transpiled_circuits
            num_parameter_sets = 1
        else:
            parameters = np.asarray(parameters, dtype=np.float64)
            if parameters.ndim == 1:
                bound_circuits = [
                    circ.bind_parameters(parameters)  # type: ignore
                    for circ in self.transpiled_circuits
                ]
                num_parameter_sets = 1
            elif parameters.ndim == 2:
                bound_circuits = [
                    circ.bind_parameters(parameter)
                    for parameter in parameters
                    for circ in self.transpiled_circuits
                ]
                num_parameter_sets = len(parameters)
            else:
                raise TypeError("The number of array dimension must be 1 or 2.")

        # Run
        run_opts = copy.copy(self.run_options)
        run_opts.update_options(**run_options)

        results = self._sampler.run(circuits=bound_circuits, **run_opts.__dict__)
        results = cast(SamplerResult, results)

        num_obs = [len(m) for _, m in self.preprocessed_circuits]
        num_experiments = num_obs * num_parameter_sets
        accum = [0] + list(accumulate(num_experiments))

        postprocessed = [
            self._postprocessing(results[accum[i] : accum[i + 1]])
            for i in range(len(num_experiments))
        ]

        return cast(
            EstimatorArrayResult,
            reduce(lambda a, b: a + b, postprocessed),
        )

    def _preprocessing(
        self, circuits: list[QuantumCircuit], observables: list[SparsePauliOp]
    ) -> list[tuple[QuantumCircuit, list[QuantumCircuit]]]:
        """
        Preprocessing for evaluation of expectation value using pauli rotation gates.
        """
        preprocessed_circuits = []
        for group in self._grouping:
            circuit = self._circuits[group.circuit_index]
            observable = self._observables[group.observable_index]
            diff_circuits: list[QuantumCircuit] = []
            if self._measurement_strategy:
                for sumop in AbelianGrouper().convert(PauliSumOp(observable)).oplist:  # type: ignore
                    op = cast(SparsePauliOp, sumop.primitive)
                    coeff_dict = {
                        key: val.real.item() if np.isreal(val) else val.item()
                        for key, val in op.label_iter()
                    }
                    lst = []
                    for paulis in zip(*coeff_dict.keys()):
                        pauli_set = set(paulis)
                        pauli_set.discard("I")
                        lst.append(pauli_set.pop() if pauli_set else "I")
                    pauli = "".join(lst)

                    meas_circuit = QuantumCircuit(circuit.num_qubits, observable.num_qubits)
                    for i, val in enumerate(reversed(pauli)):
                        if val == "Y":
                            meas_circuit.sdg(i)
                        if val in ["Y", "X"]:
                            meas_circuit.h(i)
                        meas_circuit.measure(i, i)
                    meas_circuit.metadata = {"basis": pauli, "coeff": coeff_dict}
                    diff_circuits.append(meas_circuit)
            else:
                for pauli, coeff in observable.label_iter():
                    meas_circuit = QuantumCircuit(circuit.num_qubits, observable.num_qubits)
                    for i, val in enumerate(reversed(pauli)):
                        if val == "Y":
                            meas_circuit.sdg(i)
                        if val in ["Y", "X"]:
                            meas_circuit.h(i)
                        meas_circuit.measure(i, i)
                    coeff = coeff.real.item() if np.isreal(coeff) else coeff.item()
                    meas_circuit.metadata = {"basis": pauli, "coeff": coeff}
                    diff_circuits.append(meas_circuit)

            preprocessed_circuits.append((circuit.copy(), diff_circuits))
        return preprocessed_circuits

    def _postprocessing(self, result: Union[Result, BaseResult, dict]) -> EstimatorArrayResult:
        """
        Postprocessing for evaluation of expectation value using pauli rotation gates.
        """
        if not isinstance(result, SamplerResult):
            raise TypeError(f"result must be SamplerResult, not {type(result)}.")

        data = result.counts
        metadata = result.metadata

        combined_expval = 0.0
        combined_variance = 0.0
        combined_stderr = 0.0

        for datum, meta in zip(data, metadata):
            basis = meta.get("basis", None)
            coeff = meta.get("coeff", 1)
            basis_coeff = coeff if isinstance(coeff, dict) else {basis: coeff}
            for basis, coeff in basis_coeff.items():
                diagonal = _pauli_diagonal(basis) if basis is not None else None
                # qubits = meta.get("qubits", None)
                shots = sum(datum.values())

                # Compute expval component
                expval, var = _expval_with_variance(datum, diagonal=diagonal)
                # Accumulate
                combined_expval += expval * coeff
                combined_variance += var * coeff ** 2
                combined_stderr += np.sqrt(max(var * coeff ** 2 / shots, 0.0))

        return EstimatorArrayResult(
            np.array([combined_expval], np.float64),
            np.array([combined_variance], np.float64),
        )


def _expval_with_variance(
    counts: Counts,
    diagonal: Optional[np.ndarray] = None,
    # clbits: Optional[list[int]] = None,
) -> tuple[float, float]:

    # Marginalize counts
    # if clbits is not None:
    #    counts = marginal_counts(counts, meas_qubits=clbits)

    # Get counts shots and probabilities
    probs = np.fromiter(counts.values(), dtype=float)
    shots = probs.sum()
    probs = probs / shots

    # Get diagonal operator coefficients
    if diagonal is None:
        coeffs = np.array(
            [(-1) ** (key.count("1") % 2) for key in counts.keys()], dtype=probs.dtype
        )
    else:
        keys = [int(key, 2) for key in counts.keys()]
        coeffs = np.asarray(diagonal[keys], dtype=probs.dtype)

    # Compute expval
    expval = coeffs.dot(probs)

    # Compute variance
    if diagonal is None:
        # The square of the parity diagonal is the all 1 vector
        sq_expval = np.sum(probs)
    else:
        sq_expval = (coeffs ** 2).dot(probs)
    variance = sq_expval - expval ** 2

    # Compute standard deviation
    if variance < 0:
        if not np.isclose(variance, 0):
            logger.warning(
                "Encountered a negative variance in expectation value calculation."
                "(%f). Setting standard deviation of result to 0.",
                variance,
            )
        variance = np.float64(0.0)
    return expval.item(), variance.item()


def _pauli_diagonal(pauli: str) -> np.ndarray:
    """Return diagonal for given Pauli.

    Args:
        pauli: a pauli string.

    Returns:
        np.ndarray: The diagonal vector for converting the Pauli basis
                    measurement into an expectation value.
    """
    if pauli[0] in ["+", "-"]:
        pauli = pauli[1:]

    diag = np.array([1])
    for i in reversed(pauli):
        if i == "I":
            tmp = np.array([1, 1])
        else:
            tmp = np.array([1, -1])
        diag = np.kron(tmp, diag)
    return diag
