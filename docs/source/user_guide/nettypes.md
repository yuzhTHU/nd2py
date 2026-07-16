(network-types)=
## Network types

`NetType` describes where a value lives in a network-dynamics expression:

```python
NetType = Literal["scalar", "node", "edge"]
```

It is not a NumPy shape or dtype.

| Nettype | Interpretation | Typical last dimension |
|---|---|---|
| `scalar` | One value shared across the graph | `1` or broadcastable |
| `node` | One value for each node | number of nodes |
| `edge` | One value for each edge | number of edges |

### Inference

{class}`~nd2py.core.nettype.nettype_mixin.NetTypeMixin` stores two related
pieces of state:

- `_assigned_nettypes`: hard constraints supplied by the user;
- `_possible_nettypes`: candidates remaining after inference.

`infer_nettype()` performs bottom-up and top-down propagation over the whole
expression tree. Ordinary arithmetic follows the non-scalar operand: combining
a node value with a scalar produces a node value, while directly combining
node and edge values is invalid.

```python
import nd2py as nd

x = nd.Variable("x", nettype="node")
c = nd.Number(2.0, nettype="scalar")
expression = c * x

assert expression.nettype == "node"
```

### Graph-aware operators

- `Sour(x)` and `Targ(x)` gather node values onto source or target edges.
- `Aggr(x)` and `Rgga(x)` aggregate edge values onto nodes.
- `Readout(x)` reduces a node or edge value to scalar scope.

These operators implement custom `map_nettype` rules. A custom operator can
override the same class method when the default arithmetic mapping is not
appropriate.

### Hard constraints and ambiguity

Passing `nettype=` anchors a node. If inference cannot find a compatible
combination of child types, nd2py raises a type-inference conflict rather than
silently broadcasting node and edge quantities.
