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

"""Test BindingsArray"""


import numpy as np

from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit
from qiskit.primitives import BindingsArray
from qiskit.test import QiskitTestCase


class BindingsArrayTestCase(QiskitTestCase):
    """Test the BindingsArray class"""

    def setUp(self):
        self.circuit = QuantumCircuit(5)
        self.params = ParameterVector("a", 50)
        param_iter = iter(self.params)
        for _ in range(10):
            for qubit in range(5):
                self.circuit.sx(qubit)
                self.circuit.rz(next(param_iter), qubit)
            self.circuit.cx(0, 1)
            self.circuit.cx(2, 3)
        return super().setUp()

    def test_construction_failures(self):
        """Test all the possible construction failures"""
        with self.assertRaisesRegex(ValueError, "specify a shape if no values"):
            BindingsArray()

        with self.assertRaisesRegex(ValueError, "inconsistent with last dimension of"):
            BindingsArray(kwvals={Parameter("a"): [0, 1]}, shape=())

        with self.assertRaisesRegex(ValueError, r"\(3, 5\) inconsistent with \(2,\)"):
            BindingsArray(np.empty((3, 5)), shape=2)

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            # could have shape (2,) or ()
            BindingsArray([np.empty(2), np.empty(2)])

        with self.assertRaisesRegex(ValueError, "Could not find any consistent shape"):
            BindingsArray([np.empty((5, 8, 3)), np.empty((4, 7, 2))])

        with self.assertRaisesRegex(ValueError, "inconsistent with last dimension of"):
            BindingsArray(
                vals=np.empty((5, 10)),
                kwvals={(Parameter("a"), Parameter("b")): np.empty((5, 10, 3))},
            )

    def test_bind_at_idx(self):
        """Test binding at a specified index"""
        vals = np.linspace(0, 1, 1000).reshape((5, 4, 50))
        expected_circuit = self.circuit.assign_parameters(vals[2, 3])

        ba = BindingsArray(vals)
        self.assertEqual(ba.bind(self.circuit, (2, 3)), expected_circuit)

        ba = BindingsArray([vals[:, :, :20], vals[:, :, 20:27], vals[:, :, 27:]])
        self.assertEqual(ba.bind(self.circuit, (2, 3)), expected_circuit)

        ba = BindingsArray(vals[:, :, :20], {tuple(self.params[20:]): vals[:, :, 20:]})
        self.assertEqual(ba.bind(self.circuit, (2, 3)), expected_circuit)

        order = np.arange(30, 50, dtype=int)
        np.random.default_rng().shuffle(order)
        ba = BindingsArray(
            [vals[:, :, :20], vals[:, :, 20:25]],
            {
                tuple(self.params[25:30]): vals[:, :, 25:30],
                tuple(self.params[i] for i in order): vals[:, :, order],
            },
        )
        self.assertEqual(ba.bind(self.circuit, (2, 3)), expected_circuit)

    def test_bind_all(self):
        """Test binding all possible values"""
        # this test assumes bind_all() is implemented via bind_at_idx(), which we have already
        # tested. so here, we just test that it gets the order right
        vals = np.linspace(0, 1, 300).reshape((2, 3, 50))
        bound_circuits = BindingsArray(vals).bind_all(self.circuit)
        self.assertIsInstance(bound_circuits, np.ndarray)
        self.assertEqual(bound_circuits.shape, (2, 3))
        for idx in np.ndindex((2, 3)):
            self.assertEqual(bound_circuits[idx], self.circuit.assign_parameters(vals[idx]))

    def test_properties(self):
        """Test properties"""
        with self.subTest("binding a list"):
            vals = np.linspace(0, 1, 50).tolist()
            ba = BindingsArray(vals)
            self.assertEqual(ba.num_parameters, 50)
            self.assertEqual(ba.ndim, 0)
            self.assertEqual(ba.shape, ())
            self.assertEqual(ba.size, 1)
            self.assertEqual(ba.kwvals, {})
            np.testing.assert_allclose(ba.vals, np.array(vals)[:, np.newaxis])

        with self.subTest("binding a single array"):
            vals = np.linspace(0, 1, 300).reshape((2, 3, 50))
            ba = BindingsArray(vals)
            self.assertEqual(ba.num_parameters, 50)
            self.assertEqual(ba.ndim, 2)
            self.assertEqual(ba.shape, (2, 3))
            self.assertEqual(ba.size, 6)
            self.assertEqual(ba.kwvals, {})
            np.testing.assert_allclose(ba.vals, vals.reshape((1, 2, 3, 50)))

        with self.subTest("binding multiple arrays"):
            vals = np.linspace(0, 1, 300).reshape((2, 3, 50))
            ba = BindingsArray([vals[:, :, :20], vals[:, :, 20:]])
            self.assertEqual(ba.num_parameters, 50)
            self.assertEqual(ba.ndim, 2)
            self.assertEqual(ba.shape, (2, 3))
            self.assertEqual(ba.size, 6)
            self.assertEqual(ba.kwvals, {})
            self.assertEqual(len(ba.vals), 2)
            np.testing.assert_allclose(ba.vals[0], vals[:, :, :20])
            np.testing.assert_allclose(ba.vals[1], vals[:, :, 20:])

    def test_ravel(self):
        """Test ravel"""
        vals = np.linspace(0, 1, 300).reshape((2, 3, 50))

        ba = BindingsArray(vals)
        flat = ba.ravel()
        self.assertEqual(flat.num_parameters, 50)
        self.assertEqual(flat.ndim, 1)
        self.assertEqual(flat.shape, (6,))
        self.assertEqual(flat.size, 6)
        self.assertEqual(flat.kwvals, {})
        flat_vals = vals.reshape(-1, 50)
        np.testing.assert_allclose(flat.vals, flat_vals.reshape((1, 6, 50)))

        bound_circuits = list(flat.bind_all(self.circuit).reshape(6))
        self.assertEqual(len(bound_circuits), 6)
        for i in range(6):
            self.assertEqual(bound_circuits[i], self.circuit.assign_parameters(flat_vals[i]))

    def test_reshape(self):
        """Test reshape"""
        vals = np.linspace(0, 1, 300).reshape((2, 3, 50))

        with self.subTest("reshape"):
            ba = BindingsArray(vals)
            reshape_ba = ba.reshape((3, 2))
            self.assertEqual(reshape_ba.num_parameters, 50)
            self.assertEqual(reshape_ba.ndim, 2)
            self.assertEqual(reshape_ba.shape, (3, 2))
            self.assertEqual(reshape_ba.size, 6)
            self.assertEqual(reshape_ba.kwvals, {})
            reshape_vals = vals.reshape((3, 2, 50))
            np.testing.assert_allclose(reshape_ba.vals, reshape_vals.reshape((1, 3, 2, 50)))

            circuit = self.circuit
            bound_circuits = reshape_ba.bind_all(circuit)
            self.assertEqual(bound_circuits.shape, (3, 2))
            self.assertEqual(bound_circuits[0, 0], circuit.assign_parameters(reshape_vals[0, 0]))
            self.assertEqual(bound_circuits[0, 1], circuit.assign_parameters(reshape_vals[0, 1]))
            self.assertEqual(bound_circuits[1, 0], circuit.assign_parameters(reshape_vals[1, 0]))
            self.assertEqual(bound_circuits[1, 1], circuit.assign_parameters(reshape_vals[1, 1]))
            self.assertEqual(bound_circuits[2, 0], circuit.assign_parameters(reshape_vals[2, 0]))
            self.assertEqual(bound_circuits[2, 1], circuit.assign_parameters(reshape_vals[2, 1]))

        with self.subTest("flatten"):
            ba = BindingsArray(vals)
            reshape_ba = ba.reshape(6)
            self.assertEqual(reshape_ba.num_parameters, 50)
            self.assertEqual(reshape_ba.ndim, 1)
            self.assertEqual(reshape_ba.shape, (6,))
            self.assertEqual(reshape_ba.size, 6)
            self.assertEqual(reshape_ba.kwvals, {})
            reshape_vals = vals.reshape(-1, 50)
            np.testing.assert_allclose(reshape_ba.vals, reshape_vals.reshape((1, 6, 50)))

            bound_circuits = list(reshape_ba.bind_all(self.circuit))
            self.assertEqual(len(bound_circuits), 6)
            for i in range(6):
                self.assertEqual(bound_circuits[i], self.circuit.assign_parameters(reshape_vals[i]))

    def test_kwvals(self):
        """Test constructor with kwvals"""
        with self.subTest("binding a single value"):
            vals = np.linspace(0, 1, 50)
            kwvals = {self.params: vals}
            ba = BindingsArray(kwvals=kwvals)
            self.assertEqual(ba.num_parameters, 50)
            self.assertEqual(ba.ndim, 0)
            self.assertEqual(ba.shape, ())
            self.assertEqual(ba.size, 1)
            self.assertEqual(ba.vals, [])
            self.assertEqual(ba.kwvals, {tuple(param.name for param in self.params): vals})

            bound_circuit = ba.bind(self.circuit, ())
            self.assertEqual(bound_circuit, self.circuit.assign_parameters(vals))

        with self.subTest("binding an array"):
            vals = np.linspace(0, 1, 300).reshape((2, 3, 50))
            kwvals = {self.params: vals}
            ba = BindingsArray(kwvals=kwvals)
            self.assertEqual(ba.num_parameters, 50)
            self.assertEqual(ba.ndim, 2)
            self.assertEqual(ba.shape, (2, 3))
            self.assertEqual(ba.size, 6)
            self.assertEqual(ba.vals, [])
            self.assertEqual(ba.kwvals, {tuple(param.name for param in self.params): vals})

            bound_circuits = ba.bind_all(self.circuit)
            self.assertEqual(bound_circuits.shape, (2, 3))
            self.assertEqual(bound_circuits[0, 0], self.circuit.assign_parameters(vals[0, 0]))
            self.assertEqual(bound_circuits[0, 1], self.circuit.assign_parameters(vals[0, 1]))
            self.assertEqual(bound_circuits[0, 2], self.circuit.assign_parameters(vals[0, 2]))
            self.assertEqual(bound_circuits[1, 0], self.circuit.assign_parameters(vals[1, 0]))
            self.assertEqual(bound_circuits[1, 1], self.circuit.assign_parameters(vals[1, 1]))
            self.assertEqual(bound_circuits[1, 2], self.circuit.assign_parameters(vals[1, 2]))

        with self.subTest("binding a single param"):
            vals = np.linspace(0, 1, 50)
            kwvals = {self.params[0]: vals}
            ba = BindingsArray(kwvals=kwvals)
            self.assertEqual(ba.num_parameters, 1)
            self.assertEqual(ba.ndim, 1)
            self.assertEqual(ba.shape, (50,))
            self.assertEqual(ba.size, 50)
            self.assertEqual(ba.vals, [])
            self.assertEqual(list(ba.kwvals.keys()), [(self.params[0].name,)])
            np.testing.assert_allclose(list(ba.kwvals.values()), [vals[..., np.newaxis]])

    def test_vals_kwvals(self):
        """Test constructor with vals and kwvals"""
        with self.subTest("binding a single value"):
            vals = np.linspace(0, 1, 50)
            kwvals = {tuple(self.params[20:]): vals[20:]}
            ba = BindingsArray(vals=vals[:20], kwvals=kwvals)
            self.assertEqual(ba.num_parameters, 50)
            self.assertEqual(ba.ndim, 0)
            self.assertEqual(ba.shape, ())
            self.assertEqual(ba.size, 1)
            np.testing.assert_allclose(ba.vals, vals[np.newaxis, :20])
            self.assertEqual(ba.kwvals, {tuple(p.name for p in k): v for k, v in kwvals.items()})

            bound_circuit = ba.bind(self.circuit, ())
            self.assertEqual(bound_circuit, self.circuit.assign_parameters(vals))

        with self.subTest("binding an array"):
            vals = np.linspace(0, 1, 300).reshape((2, 3, 50))
            kwvals = {tuple(self.params[20:]): vals[:, :, 20:]}
            ba = BindingsArray(vals=vals[:, :, :20], kwvals=kwvals)
            self.assertEqual(ba.num_parameters, 50)
            self.assertEqual(ba.ndim, 2)
            self.assertEqual(ba.shape, (2, 3))
            self.assertEqual(ba.size, 6)
            np.testing.assert_allclose(ba.vals, vals[np.newaxis, :, :, :20])
            self.assertEqual(ba.kwvals, {tuple(p.name for p in k): v for k, v in kwvals.items()})

            bound_circuits = ba.bind_all(self.circuit)
            self.assertEqual(bound_circuits.shape, (2, 3))
            self.assertEqual(bound_circuits[0, 0], self.circuit.assign_parameters(vals[0, 0]))
            self.assertEqual(bound_circuits[0, 1], self.circuit.assign_parameters(vals[0, 1]))
            self.assertEqual(bound_circuits[0, 2], self.circuit.assign_parameters(vals[0, 2]))
            self.assertEqual(bound_circuits[1, 0], self.circuit.assign_parameters(vals[1, 0]))
            self.assertEqual(bound_circuits[1, 1], self.circuit.assign_parameters(vals[1, 1]))
            self.assertEqual(bound_circuits[1, 2], self.circuit.assign_parameters(vals[1, 2]))

        with self.subTest("len(val) == 1 and len(kwvals) > 0"):
            ba = BindingsArray(
                vals=np.empty((5, 10)),
                kwvals={(Parameter("a"), Parameter("b")): np.empty((5, 10, 2))},
            )
            self.assertEqual(ba.num_parameters, 3)
            self.assertEqual(ba.ndim, 2)
            self.assertEqual(ba.shape, (5, 10))
            self.assertEqual(ba.size, 50)