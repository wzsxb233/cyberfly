"""Share frozen catalog metadata while keeping each request's codec private.

Conversion occurs once, after the complete original snapshot and checkpoint
validation. Every later observation still passes NeuronStateCodec.read.
"""
from copy import deepcopy
from types import MappingProxyType

from shared_io.neuron_link import NeuronStateCodec


def _freeze_json(value):
    """One independent immutable copy, made at cold verified model load."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if type(value) in (str, int, float, bool, type(None)):
        return value
    raise TypeError('Static brain catalog must contain only parsed JSON values')


class FrozenCatalogCodec(NeuronStateCodec):
    """All request state stays private; only catalog is immutable and shared.

    NeuronStateCodec.read reads catalog.macro_groups when initializing masks;
    it never writes catalog. IDs/masks, contract and any later mutable fields
    are still deep-copied. Actual observed voltages/counts remain local to read.
    """
    @classmethod
    def from_verified(cls, original):
        if isinstance(original, cls):
            return original
        if type(original) is not NeuronStateCodec or original.ids is None:
            raise ValueError('Immutable catalog requires the actual initialized, verified neuron codec')
        result = cls.__new__(cls)
        for name, value in original.__dict__.items():
            setattr(result, name, _freeze_json(value) if name == 'catalog' else deepcopy(value))
        return result

    def __deepcopy__(self, memo):
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        for name, value in self.__dict__.items():
            setattr(result, name, value if name == 'catalog' else deepcopy(value, memo))
        return result
