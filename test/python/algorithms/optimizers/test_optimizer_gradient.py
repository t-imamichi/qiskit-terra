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

""" Test Optimizer's gradient """

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

import numpy as np

from qiskit.algorithms.optimizers import Optimizer
from qiskit.utils import algorithm_globals


class TestOptimizerGradient(QiskitAlgorithmsTestCase):
    """Test Optimizer's Gradient"""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 52
        self._n = 10
        rand = algorithm_globals.random.random(2 * self._n)

        def _func(x):
            x_set = np.reshape(x, (-1, self._n))
            vals = np.array(
                [
                    sum(rand[2 * i] * np.sin(x[i] + rand[2 * i + 1]) for i in range(self._n))
                    for x in x_set
                ]
            )
            return vals[0] if len(vals) == 1 else vals

        self._func = _func

    def test_gradient(self):
        """test gradient"""
        for i in range(10):
            x = algorithm_globals.random.random(self._n)
            nd = Optimizer.gradient_num_diff(x, self._func, 1e-6, 1)
            ps = Optimizer.gradient_param_shift(x, self._func, 1)
            np.testing.assert_array_almost_equal(nd, ps)

    def test_gradient_max_evals_grouped(self):
        """test gradient with max_evals_grouped"""
        max_evals_grouped = 3
        for i in range(10):
            x = algorithm_globals.random.random(self._n)
            nd = Optimizer.gradient_num_diff(x, self._func, 1e-6, max_evals_grouped)
            ps = Optimizer.gradient_param_shift(x, self._func, 0, max_evals_grouped)
            np.testing.assert_array_almost_equal(nd, ps)


if __name__ == "__main__":
    unittest.main()
