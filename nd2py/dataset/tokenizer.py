from ..utils import AttrDict

class Tokenizer(object):
    def __init__(self):
        self.special = AttrDict(
            pad=0,
            sos=1,
            eos=2,
            query_value=3,
            query_policy=4,
            query_index=5
        )
        self.placeholder = AttrDict(node=6, edge=7)
        self.variable = AttrDict(
            node=dict(v1=10, v2=11, v3=12, v4=13, v5=14),
            edge=dict(e1=15, e2=16, e3=17, e4=18, e5=19)
        )
        self.constant = AttrDict({
            '1': 21,
            '2': 22,
            '3': 23,
            '4': 24,
            '5': 25,
            '(1/2)': 26,
            '(1/3)': 27,
            '(1/4)': 28,
            '(1/5)': 29
        })
        self.coefficient = 30
        self.operator = AttrDict(
            binary=dict(
                add=31,
                sub=32,
                mul=33,
                div=34,
                pow=35,
                # rac=36, # x^(1/y)
                regular=37
            ),
            unary=dict(
                neg=38,
                exp=39,
                logabs=40,
                sin=41,
                cos=42,
                tan=43,
                abs=44,
                inv=45,
                sqrtabs=46,
                pow2=47,
                pow3=48,
                # sinh=49,
                # cosh=50,
                tanh=51,
                sigmoid=52,
                aggr=53,
                sour=54,
                term=55
            )
        )

        self.word2id = self.special + self.placeholder + self.variable.node + self.variable.edge + self.constant + self.operator.binary + self.operator.unary
        self.id2word = {v: k for k, v in self.word2id.items()}
