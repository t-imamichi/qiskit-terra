# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
Estimator Task class
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from qiskit import QuantumCircuit

from .base_task import BaseTask
from .bindings_array import BindingsArray, BindingsArrayLike
from .dataclasses import frozen_dataclass
from .observables_array import ObservablesArray, ObservablesArrayLike
from .shape import ShapedMixin


@frozen_dataclass
class EstimatorTask(BaseTask, ShapedMixin):
    """Task for Estimator.
    Task is composed of triple (circuit, observables, parameter_values).
    """

    observables: ObservablesArray
    parameter_values: BindingsArray = BindingsArray(shape=())
    _shape: Tuple[int, ...] = ()

    def __post_init__(self):
        shape = np.broadcast_shapes(self.observables.shape, self.parameter_values.shape)
        self._shape = shape

    @classmethod
    def coerce(cls, task: EstimatorTaskLike) -> EstimatorTask:
        """Coerce EstimatorTaskLike into EstimatorTask.

        Args:
            task: an object to be estimator task.

        Returns:
            A coerced estimator task.
        """
        if isinstance(task, EstimatorTask):
            return task
        if len(task) != 2 and len(task) != 3:
            raise ValueError(f"The length of task must be 2 or 3, but length {len(task)} is given.")
        circuit = task[0]
        observables = ObservablesArray.coerce(task[1])
        if len(task) == 2:
            return cls(circuit=circuit, observables=observables)
        parameter_values = BindingsArray.coerce(task[2])
        return cls(circuit=circuit, observables=observables, parameter_values=parameter_values)

    def validate(self):
        """Validate the task."""
        super(EstimatorTask, self).validate()  # pylint: disable=super-with-arguments
        # I'm not sure why these arguments for super are needed. But if no args, tests are failed
        # for Python >=3.10. Seems to be some bug, but I can't fix.
        self.observables.validate()
        self.parameter_values.validate()
        # Cross validate circuits and observables
        for i, observable in enumerate(self.observables):
            num_qubits = len(next(iter(observable)))
            if self.circuit.num_qubits != num_qubits:
                raise ValueError(
                    f"The number of qubits of the circuit ({self.circuit.num_qubits}) does "
                    f"not match the number of qubits of the {i}-th observable ({num_qubits})."
                )
        # Cross validate circuits and paramter_values
        num_parameters = self.parameter_values.num_parameters
        if num_parameters != self.circuit.num_parameters:
            raise ValueError(
                f"The number of values ({num_parameters}) does not match "
                f"the number of parameters ({self.circuit.num_parameters}) for the circuit."
            )


EstimatorTaskLike = Union[
    EstimatorTask, Tuple[QuantumCircuit, ObservablesArrayLike, BindingsArrayLike]
]