import pytest
import nd2py as nd

@pytest.mark.parametrize("printer_cls,node,flags,expected", [
    (nd.TreePrinter, 
     nd.Variable('x') + nd.Number(1) + nd.Variable('y'), 
     {}, 
     """\
Add (scalar)
├ Add (scalar)
┆ ├ x (scalar)
┆ └ 1 (scalar)
└ y (scalar)"""),

    (nd.TreePrinter, 
     nd.Variable('x') + nd.Number(1) + nd.Variable('y'), 
     dict(flat=True), 
     """\
Add (scalar)
├ x (scalar)
├ 1 (scalar)
└ y (scalar)"""),

    (nd.TreePrinter, 
     nd.Variable('x') * nd.Number(1) * nd.Variable('y'), 
     dict(flat=True), 
     """\
Mul (scalar)
├ x (scalar)
├ 1 (scalar)
└ y (scalar)"""),

    (nd.TreePrinter, 
     nd.Variable('1') + nd.Variable('x', 'node') + nd.Aggr(nd.Variable('y', 'edge')), 
     dict(flat=True), 
     """\
Add (node)
├ 1 (scalar)
├ x (node)
└ Aggr (node)
  └ y (edge)"""),
])
def test_treeprinter_outputs(printer_cls, node, flags, expected):
    printer = printer_cls()
    assert printer(node, **flags) == expected
