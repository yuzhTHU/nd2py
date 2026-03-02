import pytest
import nd2py as nd

@pytest.mark.parametrize("printer_cls,node,expected", [
    (nd.StringPrinter, nd.Number(3.14), "3.14"),
    (nd.StringPrinter, nd.Variable("x"), "x"),
    (nd.StringPrinter, nd.Number(1) + nd.Number(2), "1 + 2"),
    (nd.StringPrinter, nd.sin(1), "sin(1)"),
    (nd.StringPrinter, nd.cos(1), "cos(1)"),
    (nd.StringPrinter, nd.tanh(1), "tanh(1)"),
    (nd.StringPrinter, nd.sigmoid(1), "sigmoid(1)"),
    (nd.StringPrinter, nd.aggr(1), "aggr(1)"),
    (nd.StringPrinter, nd.sour(1), "sour(1)"),
    (nd.StringPrinter, nd.targ(1), "targ(1)"),

])
def test_printer_outputs(printer_cls, node, expected):
    printer = printer_cls()
    assert printer(node) == expected
    assert node.to_str() == expected

def test_omit_mul_sign_and_parentheses():
    expr = nd.Number(2) * (nd.Variable('x') + nd.Variable('y'))
    sp = nd.StringPrinter()
    assert sp(expr) == "2 * (x + y)"
    assert sp(expr, omit_mul_sign=True) == "2(x + y)"

    assert expr.to_str() == "2 * (x + y)"
    assert expr.to_str(omit_mul_sign=True) == "2(x + y)"
