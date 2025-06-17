import pytest
import nd2py as nd

n = nd.Variable("n", "node")
e = nd.Variable("e", "edge")
s = nd.Variable("s", "scalar")


@pytest.mark.parametrize(
    "eq,nettype,expected",
    [
        ("n", "node", "n"),
        ("n", "edge", "targ(n)"),
        ("n", "scalar", "readout(n)"),
        ("e", "node", "aggr(e)"),
        ("e", "edge", "e"),
        ("e", "scalar", "readout(e)"),
        ("s", "node", "s"),
        ("s", "edge", "s"),
        ("s", "scalar", "s"),
        ("n+e", "node", "n + aggr(e)"),
        ("n+e", "edge", "targ(n) + e"),
        ("n+e", "scalar", "readout(n) + readout(e)"),
        ("n+s", "node", "n + s"),
        ("n+s", "edge", "targ(n + s)"),
        ("n+s", "scalar", "readout(n + s)"),
        ("e+s", "node", "aggr(e + s)"),
        ("e+s", "edge", "e + s"),
        ("e+s", "scalar", "readout(e + s)"),
        ("n*(e+e)", "node", "n * aggr(e + e)"),
        ("n*(e+e)", "edge", "targ(n) * (e + e)"),
        ("n*(e+e)", "scalar", "readout(n) * readout(e + e)"),
        ("(e+e)*n", "node", "aggr(e + e) * n"),
        ("(e+e)*n", "edge", "(e + e) * targ(n)"),
        ("(e+e)*n", "scalar", "readout(e + e) * readout(n)"),
        ("aggr(n)", "node", "aggr(targ(n))"),
        ("aggr(n)", "edge", "targ(n)"),
        ("aggr(n)", "scalar", "readout(aggr(targ(n)))"),
        ("sour(e)", "node", "aggr(e)"),
        ("sour(e)", "edge", "sour(aggr(e))"),
        ("sour(e)", "scalar", "readout(sour(aggr(e)))"),
    ],
)
def test_fix_nettype1(eq, nettype, expected):
    with nd.no_nettype_check():
        f1 = n + e
        f1 = f1.fix_nettype()
        f1.nettype
