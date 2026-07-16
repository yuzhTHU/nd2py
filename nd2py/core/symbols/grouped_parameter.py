import numpy as np
from numbers import Real
from typing import Optional, Set, Dict, Any
from .symbol import Symbol
from .variable import Variable
from ..nettype import NetType

__all__ = ["GroupedParameter"]

class GroupedParameter(Symbol):
    n_operands = 1
    
    def __init__(
        self, 
        *operands, 
        value: Optional[Dict[Any, float]] = None,
        default: Optional[float] = None,
        fitable: bool = True,
        nettype: Optional[NetType|Set[NetType]] = None, 
    ):
        """Create a fitable parameter lookup indexed by a categorical variable.

        Each distinct value of the Variable operand selects one scalar entry
        of :attr:`value`. Labels not present in ``value`` are added on first
        evaluation and initialized with ``default``.

        Args:
            *operands: Exactly one categorical :class:`Variable` used to
                select parameter values.
            value: Optional initial mapping from category labels to numerical
                parameter values. Mapping order is preserved.
            default: Value assigned to a previously unseen label. ``None`` is
                interpreted as ``0.0``.
            fitable: Whether :class:`~nd2py.BFGSFit` may optimize the values.
            nettype: Optional network-type constraint. By default, the result
                follows the Variable operand.

        Examples:
            >>> import numpy as np
            >>> import nd2py as nd
            >>> s = nd.Variable("s")
            >>> parameter = nd.GroupedParameter(s, default=1.0)
            >>> parameter.eval({"s": np.array(["a", "b", "a"])})
            array([1., 1., 1.])
            >>> parameter.value_dict
            {'a': 1.0, 'b': 1.0}
        """
        if len(operands) != 1 or not isinstance(operands[0], Variable):
            raise TypeError("GroupedParameter expects exactly one Variable operand")
        if value is not None and not isinstance(value, dict):
            raise TypeError("value must be a mapping from group labels to numbers")
        if value is not None and not all(isinstance(v, Real) for v in value.values()):
            raise TypeError("all grouped parameter values must be real numbers")
        super().__init__(*operands, nettype=nettype)
        if value is None:
            group_labels = []
            value = np.empty(0, dtype=float)
        else:
            group_labels = list(value.keys())
            value = np.asarray(list(value.values()), dtype=float)

        self.value = value
        self.default = 0.0 if default is None else float(default)
        self.fitable = fitable
        self.group_labels = group_labels
        self.label_to_index = {label: idx for idx, label in enumerate(group_labels)}

    @property
    def by(self) -> Variable:
        return self.operands[0]

    def bind(self, labels) -> None:
        """Add previously unseen labels initialized with :attr:`default`.

        Labels are appended in first-seen order. Existing labels and parameter
        values are left unchanged.

        Args:
            labels: Scalar or array-like categorical labels.
        """
        new_labels = []
        for label in np.asarray(labels, dtype=object).reshape(-1):
            # Convert NumPy scalar keys to their Python equivalents when possible.
            label = label.item() if isinstance(label, np.generic) else label
            if label not in self.label_to_index:
                self.label_to_index[label] = len(self.group_labels) + len(new_labels)
                new_labels.append(label)
        if new_labels:
            self.group_labels.extend(new_labels)
            self.value = np.concatenate([
                np.asarray(self.value, dtype=float), 
                np.full(len(new_labels), self.default)
            ])

    @property
    def value_dict(self):
        """Return the current label-to-parameter mapping."""
        return dict(zip(self.group_labels, np.asarray(self.value).tolist()))
