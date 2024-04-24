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
Base Pub class
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .bit_array import BitArray
from .data_bin import DataBin


class PubResult:
    """Result of Primitive Unified Bloc."""

    __slots__ = ("_data", "_metadata")

    def __init__(self, data: DataBin, metadata: dict[str, Any] | None = None):
        """Initialize a pub result.

        Args:
            data: Result data.
            metadata: Metadata specific to this pub. Keys are expected to be strings.
        """
        self._data = data
        self._metadata = metadata or {}

    def __repr__(self):
        metadata = f", metadata={self.metadata}" if self.metadata else ""
        return f"{type(self).__name__}(data={self._data}{metadata})"

    @property
    def data(self) -> DataBin:
        """Result data for the pub."""
        return self._data

    @property
    def metadata(self) -> dict:
        """Metadata for the pub."""
        return self._metadata

    def join_data(self, names: Iterable[str] | None = None) -> BitArray | np.ndarray:
        """Join data from many registers into one data container.

        Data is joined along the bits axis. For example, for :class:`~.BitArray` data, this corresponds
        to bitstring concatenation.

        Args:
            names: Which registers to join. Their order is maintained, for example, given
                ``["alpha", "beta"]``, the data from register ``alpha`` is placed to the left of the
                data from register ``beta``. When ``None`` is given, this value is set to the
                ordered list of register names, which will have been preserved from the input circuit
                order.

        Returns:
            Joint data.

        Raises:
            ValueError: If specified names are empty.
            ValueError: If specified name does not exist.
            TypeError: If specified data comes from incompatible types.
        """
        if names is None:
            names = list(self.data)
            if not names:
                raise ValueError("No entry exists in the data bin.")
        else:
            if not names:
                raise ValueError("An empty name list is given.")
            for name in names:
                if name not in self.data:
                    raise ValueError(f"Name {name} does not exist.")

        data = [self.data[name] for name in names]
        if isinstance(data[0], BitArray):
            if not all(isinstance(datum, BitArray) for datum in data):
                raise TypeError("Data comes from incompatible types.")
            joint_data = BitArray.stack_bits(data)
        elif isinstance(data[0], np.ndarray):
            if not all(isinstance(datum, np.ndarray) for datum in data):
                raise TypeError("Data comes from incompatible types.")
            joint_data = np.concatenate(data, axis=-1)
        else:
            raise TypeError("Data comes from incompatible types.")
        return joint_data
