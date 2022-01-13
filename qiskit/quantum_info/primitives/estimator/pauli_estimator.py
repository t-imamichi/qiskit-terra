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

import logging
from typing import Optional, Union, cast

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.opflow import AbelianGrouper, PauliSumOp
from qiskit.providers import BackendV1 as Backend
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result import Counts

from ..sampler import BaseSampler
from ..results import EstimatorResult, SamplerResult
from .base_estimator import BaseEstimator

logger = logging.getLogger(__name__)


class PauliEstimator(BaseEstimator):
    """
    Evaluates expectation value using pauli rotation gates.
    """

    def __init__(
        self,
        circuit: Union[QuantumCircuit, Statevector],
        observable: Union[BaseOperator, PauliSumOp],
        backend: Union[Backend, BaseSampler],
        grouping: bool = True,
    ):
        super().__init__(
            circuit=circuit,
            observable=observable,
            backend=BaseSampler.from_backend(backend),
        )
        self._grouping = grouping

    def _preprocessing(
        self, circuit: QuantumCircuit, observable: SparsePauliOp
    ) -> Union[list[QuantumCircuit], tuple[QuantumCircuit, list[QuantumCircuit]]]:
        """
        Preprocessing for evaluation of expectation value using pauli rotation gates.
        """
        diff_circuits: list[QuantumCircuit] = []
        if self._grouping:
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

        return circuit.copy(), diff_circuits

    def _postprocessing(self, result: Union[SamplerResult, dict]) -> EstimatorResult:
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

        return EstimatorResult(
            combined_expval,
            combined_variance,
            (combined_expval - combined_stderr, combined_expval + combined_stderr),
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
