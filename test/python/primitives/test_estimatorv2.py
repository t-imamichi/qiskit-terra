# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Estimator."""

import unittest

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import BindingsArray, EstimatorTask, ObservablesArray
from qiskit.primitives.statevector_estimator import Estimator, Options
from qiskit.providers import JobV1
from qiskit.quantum_info import SparsePauliOp
from qiskit.test import QiskitTestCase


class TestEstimatorV2(QiskitTestCase):
    """Test Estimator"""

    def setUp(self):
        super().setUp()
        self.ansatz = RealAmplitudes(num_qubits=2, reps=2)
        self.observable = SparsePauliOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )
        self.expvals = -1.0284380963435145, -1.284366511861733

        self.psi = (RealAmplitudes(num_qubits=2, reps=2), RealAmplitudes(num_qubits=2, reps=3))
        self.params = tuple(psi.parameters for psi in self.psi)
        self.hamiltonian = (
            SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)]),
            SparsePauliOp.from_list([("IZ", 1)]),
            SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)]),
        )
        self.theta = (
            [0, 1, 1, 2, 3, 5],
            [0, 1, 1, 2, 3, 5, 8, 13],
            [1, 2, 3, 4, 5, 6],
        )

    def test_estimator_run(self):
        """Test Estimator.run()"""
        psi1, psi2 = self.psi
        hamiltonian1, hamiltonian2, hamiltonian3 = self.hamiltonian
        theta1, theta2, theta3 = self.theta
        estimator = Estimator()

        # Specify the circuit and observable by indices.
        # calculate [ <psi1(theta1)|H1|psi1(theta1)> ]
        job = estimator.run([(psi1, hamiltonian1, [theta1])])
        self.assertIsInstance(job, JobV1)
        result = job.result()
        np.testing.assert_allclose(result[0].data.evs, [1.5555572817900956])

        # Objects can be passed instead of indices.
        # Note that passing objects has an overhead
        # since the corresponding indices need to be searched.
        # User can append a circuit and observable.
        # calculate [ <psi2(theta2)|H2|psi2(theta2)> ]
        result2 = estimator.run([(psi2, hamiltonian1, theta2)]).result()
        np.testing.assert_allclose(result2[0].data.evs, [2.97797666])

        # calculate [ <psi1(theta1)|H2|psi1(theta1)>, <psi1(theta1)|H3|psi1(theta1)> ]
        result3 = estimator.run([(psi1, [hamiltonian2, hamiltonian3], theta1)]).result()
        np.testing.assert_allclose(result3[0].data.evs, [-0.551653, 0.07535239])

        # calculate [ [<psi1(theta1)|H1|psi1(theta1)>,
        #              <psi1(theta3)|H3|psi1(theta3)>],
        #             [<psi2(theta2)|H2|psi2(theta2)>] ]
        result4 = estimator.run(
            [(psi1, [hamiltonian1, hamiltonian3], [theta1, theta3]), (psi2, hamiltonian2, theta2)]
        ).result()
        np.testing.assert_allclose(result4[0].data.evs, [1.55555728, -1.08766318])
        np.testing.assert_allclose(result4[1].data.evs, [0.17849238])

    def test_estimator_with_task(self):
        """Test estimator with explicit EstimatorTask."""
        psi1, psi2 = self.psi
        hamiltonian1, hamiltonian2, hamiltonian3 = self.hamiltonian
        theta1, theta2, theta3 = self.theta

        obs1 = ObservablesArray.coerce([hamiltonian1, hamiltonian3])
        bind1 = BindingsArray.coerce([theta1, theta3])
        task1 = EstimatorTask(psi1, obs1, bind1)
        obs2 = ObservablesArray.coerce(hamiltonian2)
        bind2 = BindingsArray.coerce(theta2)
        task2 = EstimatorTask(psi2, obs2, bind2)

        estimator = Estimator()
        result4 = estimator.run([task1, task2]).result()
        np.testing.assert_allclose(result4[0].data.evs, [1.55555728, -1.08766318])
        np.testing.assert_allclose(result4[1].data.evs, [0.17849238])

    def test_estimator_run_no_params(self):
        """test for estimator without parameters"""
        circuit = self.ansatz.assign_parameters([0, 1, 1, 2, 3, 5])
        est = Estimator()
        result = est.run((circuit, self.observable)).result()
        np.testing.assert_allclose(result[0].data.evs, [-1.284366511861733])

    def test_run_single_circuit_observable(self):
        """Test for single circuit and single observable case."""
        est = Estimator()

        with self.subTest("No parameter"):
            qc = QuantumCircuit(1)
            qc.x(0)
            op = SparsePauliOp("Z")
            param_vals = [None, [], [[]], np.array([]), np.array([[]]), [np.array([])]]
            target = [-1]
            for val in param_vals:
                self.subTest(f"{val}")
                result = est.run((qc, op, val)).result()
                np.testing.assert_allclose(result[0].data.evs, target)
                self.assertIsNone(result[0].metadata["shots"])

        with self.subTest("One parameter"):
            param = Parameter("x")
            qc = QuantumCircuit(1)
            qc.ry(param, 0)
            op = SparsePauliOp("Z")
            param_vals = [
                [np.pi],
                [[np.pi]],
                np.array([np.pi]),
                np.array([[np.pi]]),
                [np.array([np.pi])],
            ]
            target = [-1]
            for val in param_vals:
                self.subTest(f"{val}")
                result = est.run((qc, op, val)).result()
                np.testing.assert_allclose(result[0].data.evs, target)
                self.assertIsNone(result[0].metadata["shots"])

        with self.subTest("More than one parameter"):
            qc = self.psi[0]
            op = self.hamiltonian[0]
            param_vals = [
                self.theta[0],
                [self.theta[0]],
                np.array(self.theta[0]),
                np.array([self.theta[0]]),
                [np.array(self.theta[0])],
            ]
            target = [1.5555572817900956]
            for val in param_vals:
                self.subTest(f"{val}")
                result = est.run((qc, op, val)).result()
                np.testing.assert_allclose(result[0].data.evs, target)
                self.assertIsNone(result[0].metadata["shots"])

    def test_run_1qubit(self):
        """Test for 1-qubit cases"""
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(1)
        qc2.x(0)

        op = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("Z", 1)])

        est = Estimator()
        result = est.run((qc, op)).result()
        np.testing.assert_allclose(result[0].data.evs, [1])

        result = est.run((qc, op2)).result()
        np.testing.assert_allclose(result[0].data.evs, [1])

        result = est.run((qc2, op)).result()
        np.testing.assert_allclose(result[0].data.evs, [1])

        result = est.run((qc2, op2)).result()
        np.testing.assert_allclose(result[0].data.evs, [-1])

    def test_run_2qubits(self):
        """Test for 2-qubit cases (to check endian)"""
        qc = QuantumCircuit(2)
        qc2 = QuantumCircuit(2)
        qc2.x(0)

        op = SparsePauliOp.from_list([("II", 1)])
        op2 = SparsePauliOp.from_list([("ZI", 1)])
        op3 = SparsePauliOp.from_list([("IZ", 1)])

        est = Estimator()
        result = est.run((qc, op)).result()
        np.testing.assert_allclose(result[0].data.evs, [1])

        result = est.run((qc2, op)).result()
        np.testing.assert_allclose(result[0].data.evs, [1])

        result = est.run((qc, op2)).result()
        np.testing.assert_allclose(result[0].data.evs, [1])

        result = est.run((qc2, op2)).result()
        np.testing.assert_allclose(result[0].data.evs, [1])

        result = est.run((qc, op3)).result()
        np.testing.assert_allclose(result[0].data.evs, [1])

        result = est.run((qc2, op3)).result()
        np.testing.assert_allclose(result[0].data.evs, [-1])

    def test_run_errors(self):
        """Test for errors"""
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(2)

        op = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("II", 1)])

        est = Estimator()
        # TODO: add validation
        with self.assertRaises(ValueError):
            est.run((qc, op2)).result()
        with self.assertRaises(ValueError):
            est.run((qc, op, [[1e4]])).result()
        with self.assertRaises(ValueError):
            est.run((qc2, op2, [[1, 2]])).result()
        with self.assertRaises(ValueError):
            est.run((qc, [op, op2], [[1]])).result()

    def test_run_numpy_params(self):
        """Test for numpy array as parameter values"""
        qc = RealAmplitudes(num_qubits=2, reps=2)
        op = SparsePauliOp.from_list([("IZ", 1), ("XI", 2), ("ZY", -1)])
        k = 5
        params_array = np.random.rand(k, qc.num_parameters)
        params_list = params_array.tolist()
        params_list_array = list(params_array)
        estimator = Estimator()
        target = estimator.run((qc, op, params_list)).result()

        with self.subTest("ndarrary"):
            result = estimator.run((qc, op, params_array)).result()
            self.assertEqual(len(result[0].data.evs), k)
            np.testing.assert_allclose(result[0].data.evs, target[0].data.evs)

        with self.subTest("list of ndarray"):
            result = estimator.run((qc, op, params_list_array)).result()
            self.assertEqual(len(result[0].data.evs), k)
            np.testing.assert_allclose(result[0].data.evs, target[0].data.evs)

    def test_run_with_shots_option(self):
        """test with shots option."""
        est = Estimator(options={"execution": {"shots": 1024, "seed": 15}})
        result = est.run((self.ansatz, self.observable, [[0, 1, 1, 2, 3, 5]])).result()
        np.testing.assert_allclose(result[0].data.evs, [-1.307397243478641])
        self.assertEqual(result[0].metadata["shots"], 1024)

    def test_run_with_shots_option_none(self):
        """test with shots=None option. Seed is ignored then."""
        est = Estimator(options={"execution": {"shots": None, "seed": 42}})
        result_42 = est.run((self.ansatz, self.observable, [[0, 1, 1, 2, 3, 5]])).result()
        est.options.execution.seed = 15
        result_15 = est.run((self.ansatz, self.observable, [[0, 1, 1, 2, 3, 5]])).result()
        np.testing.assert_allclose(result_42[0].data.evs, result_15[0].data.evs)

    def test_options(self):
        """Test for options"""
        with self.subTest("init"):
            estimator = Estimator(options={"execution": {"shots": 3000}})
            self.assertEqual(estimator.options.execution.shots, 3000)
        with self.subTest("set_options"):
            estimator.options.execution.shots = 1024
            estimator.options.execution.seed = 15
            self.assertEqual(estimator.options.execution.shots, 1024)
            self.assertEqual(estimator.options.execution.seed, 15)
        with self.subTest("run"):
            result = estimator.run((self.ansatz, self.observable, [[0, 1, 1, 2, 3, 5]])).result()
            np.testing.assert_allclose(result[0].data.evs, [-1.307397243478641])
            self.assertEqual(result[0].metadata["shots"], 1024)
        with self.subTest("Options class"):
            options = Options()
            options.execution.shots = 1024  # pylint: disable=assigning-non-slot # pylint's bug?
            options.execution.seed = 15  # pylint: disable=assigning-non-slot
            estimator = Estimator(options=options)
            result = estimator.run((self.ansatz, self.observable, [[0, 1, 1, 2, 3, 5]])).result()
            np.testing.assert_allclose(result[0].data.evs, [-1.307397243478641])
            self.assertEqual(result[0].metadata["shots"], 1024)

    def test_negative_variance(self):
        """Test for negative variance caused by numerical error."""
        qc = QuantumCircuit(1)

        estimator = Estimator(options={"execution": {"shots": 1024}})
        result = estimator.run((qc, 1e-4 * SparsePauliOp("I"))).result()
        self.assertEqual(result[0].data.evs, 1e-4)
        self.assertEqual(result[0].data.stds, 0.0)


if __name__ == "__main__":
    unittest.main()