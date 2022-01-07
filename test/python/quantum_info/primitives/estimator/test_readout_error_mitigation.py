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

"""Tests for ReadoutErrorMitigation."""

import unittest

from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.primitives import PauliEstimator
from qiskit.quantum_info.primitives.backends import ReadoutErrorMitigation
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeBogota
from qiskit.utils import has_aer

if has_aer():
    from qiskit import Aer


class TestReadoutErrorMitigation(QiskitTestCase):
    """Test ReadoutErrorMitigation"""

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

    @unittest.skipUnless(has_aer(), "qiskit-aer doesn't appear to be installed.")
    def test_tensored_mitigation(self):
        """test for tensored mitigation"""
        backend = Aer.get_backend("aer_simulator").from_backend(FakeBogota())
        backend.set_options(seed_simulator=15)
        mit_tensored = ReadoutErrorMitigation(
            backend,
            mitigation="tensored",
            refresh=600,
            shots=1000,
            mit_pattern=[[0], [1]],
        )
        with PauliEstimator(self.ansatz, self.observable, backend=mit_tensored) as est:
            est.set_transpile_options(seed_transpiler=15)
            est.set_run_options(seed_simulator=15)
            result = est([0, 1, 1, 2, 3, 5])
        self.assertAlmostEqual(result.value, -1.30857452503)

    @unittest.skipUnless(has_aer(), "qiskit-aer doesn't appear to be installed.")
    def test_mthree_mitigation(self):
        """tett for mthree mitigator"""
        backend = Aer.get_backend("aer_simulator").from_backend(FakeBogota())
        backend.set_options(seed_simulator=15)
        mit_mthree = ReadoutErrorMitigation(
            backend,
            mitigation="mthree",
            refresh=600,
            shots=1000,
            qubits=[0, 1],
        )
        with PauliEstimator(self.ansatz, self.observable, backend=mit_mthree) as est:
            est.set_transpile_options(seed_transpiler=15)
            est.set_run_options(seed_simulator=15)
            result = est([0, 1, 1, 2, 3, 5])
        self.assertAlmostEqual(result.value, -1.3032236892)
