## Expression trees

`Symbol` inherits tree operations from
{class}`~nd2py.core.tree.tree_mixin.TreeMixin`. These operations work for built-in
and custom Symbols because they depend on `operands` and `parent`, not on a
fixed operator list.

### Traversal

```python
import nd2py as nd

x = nd.Variable("x")
expression = nd.sin(2 * x + 1)

for node in expression.iter_preorder():
    print(type(node).__name__)
# Sin
# Add
# Mul
# Number
# Variable
# Number

for node in expression.iter_postorder():
    print(type(node).__name__)
# Number
# Variable
# Mul
# Number
# Add
# Sin
```

Preorder visits a node before its children. Postorder visits children before
their parent and is useful for bottom-up analysis.

### Paths and replacement

```python
subexpression = next(
    node for node in expression.iter_preorder()
    if isinstance(node, nd.Variable)
)
path = expression.path_to(subexpression)
same_node = expression.get_path(path)

print(path)
print(same_node)
# (0, 0, 1)
# x

replacement = nd.Variable("y")
expression = expression.replace(subexpression, replacement)
print(expression)
# sin(2 * y + 1)
```

Replacement must begin at a root expression. When the root itself is replaced,
use the returned value because Python references to the old root cannot be
updated automatically.

### Copying

```python
copied = expression.copy()
```

Copies have independent parent links and parameter values. Stateful Symbol
types may need a specialized `GetCopy.visit_<Type>` implementation; stateless
operators whose constructor accepts only operands and `nettype` use the generic
implementation.

### Structural matching

Variables in a pattern act as placeholders:

```python
a = nd.Variable("a")
x = nd.Variable("x")

target = nd.sin(2 * x + 1)
bindings = target.match(nd.sin(a))

print(bindings)
# {'a': 2 * x + 1}
```

Repeated occurrences of the same pattern variable must bind to the same
subexpression.
