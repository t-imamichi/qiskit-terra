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

"""Tests for Gradient."""

import unittest
from test import combine

import numpy as np
from ddt import ddt

from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.primitives import (
    ExactEstimator,
    FiniteDiffGradient,
    OpflowGradient,
    ParameterShiftGradient,
)
from qiskit.quantum_info.primitives.results import EstimatorGradientResult
from qiskit.test import QiskitTestCase
from qiskit.utils import has_aer

if has_aer():
    from qiskit import Aer


@ddt
class TestGradient(QiskitTestCase):
    """Test Gradient"""

    def setUp(self):
        super().setUp()
        self.ansatz = RealAmplitudes(num_qubits=2, reps=2)
        self.observable = PauliSumOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )
        self.parameters = [0, 1, 1, 2, 3, 5]

        self.ansatz2 = self.ansatz.decompose()
        a = Parameter("a")
        self.ansatz2.rx(a * a, 0)
        self.ansatz2.cry(2 * a + 1, 1, 0)
        self.ansatz2.rzz(-a, 0, 1)

    @unittest.skipUnless(has_aer(), "qiskit-aer doesn't appear to be installed.")
    def test_fin_diff(self):
        """test for FiniteDiffGradient"""
        backend = Aer.get_backend("aer_simulator")
        with ExactEstimator([self.ansatz], [self.observable], backend=backend) as est:
            with FiniteDiffGradient(est, 1e-8) as grad:
                result = grad(self.parameters)
        self.assertIsInstance(result, EstimatorGradientResult)
        np.testing.assert_allclose(
            result.values,
            [0.28213347, 0.42656751, 0.20442583, 0.42656749, -0.17291453, 0.0589814],
            rtol=1e-6,
        )

    @unittest.skipUnless(has_aer(), "qiskit-aer doesn't appear to be installed.")
    def test_param_shift(self):
        """test for ParameterShiftGradient"""
        backend = Aer.get_backend("aer_simulator")
        with ExactEstimator([self.ansatz], [self.observable], backend=backend) as est:
            with ParameterShiftGradient(est) as grad:
                result = grad(self.parameters)
        self.assertIsInstance(result, EstimatorGradientResult)
        np.testing.assert_allclose(
            result.values,
            [0.28213347, 0.42656751, 0.20442583, 0.42656749, -0.17291453, 0.0589814],
            rtol=1e-6,
        )

    @unittest.skipUnless(has_aer(), "qiskit-aer doesn't appear to be installed.")
    @combine(method=["param_shift", "lin_comb", "fin_diff"])
    def test_opflow_gradient(self, method):
        """test for OpflowGradient"""
        backend = Aer.get_backend("aer_simulator")
        with ExactEstimator([self.ansatz2], [self.observable], backend=backend) as est:
            est.set_run_options(seed_simulator=15)
            x = [-1, 0, 1, 1, 2, 3, 5]
            x2 = [0, 1, 0, 2, 0, 3, 0]
            with OpflowGradient(est, method) as grad:
                result = grad([x, x2], shots=10000)
        np.testing.assert_allclose(
            result.values,
            [
                [
                    -0.51173917,
                    0.19417395,
                    0.58446423,
                    0.07498692,
                    0.58446423,
                    -0.22958507,
                    0.11165073,
                ],
                [
                    -0.39488471,
                    -0.03160609,
                    0.38505095,
                    0.43984585,
                    -0.16207361,
                    -0.03160609,
                    -0.13149839,
                ],
            ],
            rtol=1e-6,
        )
