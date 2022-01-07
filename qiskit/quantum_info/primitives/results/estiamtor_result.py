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
Expectation value result class
"""

from __future__ import annotations

from typing import Optional

from .base_result import BaseResult


class EstimatorResult(BaseResult):
    """
    Result of ExpectationValue
    #TODO doc
    """

    value: float
    variance: Optional[float] = None
    confidence_interval: Optional[tuple[float, float]] = None
    raw_data: Optional[dict] = None
    # metadata: Metadata
