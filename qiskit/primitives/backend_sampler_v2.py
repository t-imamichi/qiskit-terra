# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Sampler implementation for an arbitrary Backend object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import warnings

import numpy as np
from numpy.typing import NDArray

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.result import Result
from qiskit.transpiler.passmanager import PassManager

from qiskit.primitives.backend_estimator import _run_circuits
from qiskit.primitives.base import BaseSamplerV2
from qiskit.primitives.containers import (
    BitArray,
    PrimitiveResult,
    PubResult,
    SamplerPubLike,
    make_data_bin,
)
from qiskit.primitives.containers.bit_array import _min_num_bytes
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.primitives.containers.sampler_pub import SamplerPub


@dataclass
class ExecutionOptions:
    """Options for execution."""

    shots: int = 102
    seed: int | np.random.Generator | None = None


@dataclass
class Options:
    """Options for the primitives.

    Args:
        execution: Execution time options. See :class:`ExecutionOptions` for all available options.
    """

    execution: ExecutionOptions = field(default_factory=ExecutionOptions)
    transpilation: dict[str, Any] = field(default_factory=dict)


@dataclass
class _MeasureInfo:
    creg_name: str
    num_bits: int
    num_bytes: int
    start: int


class BackendSampler(BaseSamplerV2):
    """A :class:`~.BaseSampler` implementation that provides an interface for
    leveraging the sampler interface from any backend.

    This class provides a sampler interface from any backend and doesn't do
    any measurement mitigation, it just computes the probability distribution
    from the counts. It facilitates using backends that do not provide a
    native :class:`~.BaseSampler` implementation in places that work with
    :class:`~.BaseSampler`, such as algorithms in :mod:`qiskit.algorithms`
    including :class:`~.qiskit.algorithms.minimum_eigensolvers.SamplingVQE`.
    However, if you're using a provider that has a native implementation of
    :class:`~.BaseSampler`, it is a better choice to leverage that native
    implementation as it will likely include additional optimizations and be
    a more efficient implementation. The generic nature of this class
    precludes doing any provider- or backend-specific optimizations.
    """

    def __init__(
        self,
        *,
        backend: BackendV1 | BackendV2,
        options: Options | None = None,
        pass_manager: PassManager | None = None,
        skip_transpilation: bool = False,
    ):
        """Initialize a new BackendSampler

        Args:
            backend: Required: the backend to run the sampler primitive on
            options: Default options.
            pass_manager: An optional pass manager to use for the internal compilation
            skip_transpilation: If this is set to True the internal compilation
                of the input circuits is skipped and the circuit objects
                will be directly executed when this objected is called.
        Raises:
            ValueError: If backend is not provided
        """
        super().__init__()
        if options is None:
            self.options = Options()
        elif not isinstance(options, Options):
            self.options = Options(**options)
        self._backend = backend
        self._circuits = []
        self._parameters = []
        self._transpile_options = Options()
        self._pass_manager = pass_manager
        self._skip_transpilation = skip_transpilation

    @property
    def backend(self) -> BackendV1 | BackendV2:
        """
        Returns:
            The backend which this sampler object based on
        """
        return self._backend

    @property
    def transpile_options(self) -> Dict[str, Any]:
        """Return the transpiler options for transpiling the circuits."""
        return self.options.transpilation

    def set_transpile_options(self, **fields):
        """Set the transpiler options for transpiler.
        Args:
            **fields: The fields to update the options.
        Returns:
            self.
        """
        self.options.transpilation.update(**fields)

    def _transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        if self._skip_transpilation:
            ret = circuit
        elif self._pass_manager:
            ret = self._pass_manager.run(circuit)
        else:
            from qiskit.compiler import transpile

            ret = transpile(
                circuit,
                self.backend,
                **self.options.transpilation,
            )
        return ret

    def run(self, pubs: Iterable[SamplerPubLike], *, shots: int | None = None) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        if shots is None:
            shots = self.options.execution.shots
        coerced_pubs = [SamplerPub.coerce(pub, shots) for pub in pubs]
        if any(len(pub.circuit.cregs) == 0 for pub in coerced_pubs):
            warnings.warn(
                "One of your circuits has no output classical registers and so the result "
                "will be empty. Did you mean to add measurement instructions?",
                UserWarning,
            )
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _run(self, pubs: Iterable[SamplerPub]) -> PrimitiveResult[PubResult]:
        results = [self._run_pub(pub) for pub in pubs]
        return PrimitiveResult(results)
    
    def _run_pub(self, pub: SamplerPub) -> PubResult:
        circuit = self._transpile(pub.circuit)
        meas_info, max_num_bytes = _analyze_circuit(pub.circuit)
        bound_circuits = pub.parameter_values.bind_all(circuit)
        arrays = {
            item.creg_name: np.zeros(
                bound_circuits.shape + (pub.shots, item.num_bytes), dtype=np.uint8
            )
            for item in meas_info
        }
        flatten_circuits = np.ravel(bound_circuits).tolist()
        result_memory, _ = _run_circuits(
            flatten_circuits, self._backend, memory=True, shots=pub.shots
            #flatten_circuits, self._backend, memory=True, **self.options.execution.__dict__
        )
        memory_list = _prepare_memory(result_memory, max_num_bytes)

        for samples, index in zip(memory_list, np.ndindex(*bound_circuits.shape)):
            for item in meas_info:
                ary = _samples_to_packed_array(samples, item.num_bits, item.start)
                arrays[item.creg_name][index] = ary

        data_bin_cls = make_data_bin(
            [(item.creg_name, BitArray) for item in meas_info],
            shape=bound_circuits.shape,
        )
        meas = {
            item.creg_name: BitArray(arrays[item.creg_name], item.num_bits)
            for item in meas_info
        }
        data_bin = data_bin_cls(**meas)
        return PubResult(data_bin, metadata={"shots": pub.shots})


def _analyze_circuit(circuit: QuantumCircuit) -> Tuple[List[_MeasureInfo], int]:
    meas_info = []
    start = 0
    for creg in circuit.cregs:
        name = creg.name
        num_bits = creg.size
        meas_info.append(
            _MeasureInfo(
                creg_name=name,
                num_bits=num_bits,
                num_bytes=_min_num_bytes(num_bits),
                start=start,
            )
        )
        start += num_bits
    return meas_info, _min_num_bytes(start)


def _prepare_memory(results: List[Result], num_bytes: int) -> NDArray[np.uint8]:
    lst = []
    for res in results:
        for exp in res.results:
            if hasattr(exp.data, "memory"):
                data = b"".join(int(i, 16).to_bytes(num_bytes, "big") for i in exp.data.memory)
                data = np.frombuffer(data, dtype=np.uint8).reshape(-1, num_bytes)
            else:
                # no measure in a circuit
                data = np.zeros((exp.shots, num_bytes), dtype=np.uint8)
            lst.append(data)
    ary = np.array(lst, copy=False)
    return np.unpackbits(ary, axis=-1, bitorder="big")


def _samples_to_packed_array(
    samples: NDArray[np.uint8], num_bits: int, start: int
) -> NDArray[np.uint8]:
    # samples of `Backend.run(memory=True)` will be the order of
    # clbit_last, ..., clbit_1, clbit_0
    # place samples in the order of clbit_start+num_bits-1, ..., clbit_start+1, clbit_start
    if start == 0:
        ary = samples[:, -start - num_bits :]
    else:
        ary = samples[:, -start - num_bits : -start]
    # pad 0 in the left to align the number to be mod 8
    # since np.packbits(bitorder='big') pads 0 to the right.
    pad_size = -num_bits % 8
    ary = np.pad(ary, ((0, 0), (pad_size, 0)), constant_values=0)
    # pack bits in big endian order
    ary = np.packbits(ary, axis=-1, bitorder="big")
    return ary
