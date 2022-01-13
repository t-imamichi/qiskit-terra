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
"""
Sampler result class
"""

from __future__ import annotations

from qiskit.result import Counts, Result
from .base_result import BaseResult


class SamplerResult(BaseResult):
    """
    Result of Sampler
    """

    counts: list[Counts]
    shots: int
    raw_results: list[Result]
    metadata: list[dict]

    def __getitem__(self, key):
        return SamplerResult(self.counts[key], self.shots, self.raw_results, self.metadata[key])
