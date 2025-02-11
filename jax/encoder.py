import jax.numpy as jnp
from jax import random
from attention import MultiHeadedSelfAttention
from ffn import FFN
from layernorm import Layernorm
from dropout import dropout

class Encoder:
    def __init__(self, key=random.key(0), stack_size=6, dropout_p=0.1, dtype=jnp.float32):
        self.n = stack_size
        self.keys = random.split(key, 6*self.n)
        self.stacks = [(
            MultiHeadedSelfAttention(key=keys[6*i], dtype=dtype),
            Layernorm(key=keys[6*i+1], dtype=dtype),
            FFN(key=keys[6*i+2], dtype=dtype),
            Layernorm(key=keys[6*i+3], dtype=dtype)
        ) for i in range(self.n)]
        self.dropout_p = dropout_p

    def __call__(self, x, train=True):
        y = x
        for i, (mhsa, lnorm1, ffn, lnorm2) in enumerate(self.stacks):
            y = y + dropout(mhsa(y), self.dropout_p, key=self.keys[6*i+4], train=train)
            y = lnorm1(y)
            y = y + dropout(ffn(y), self.dropout_p, key=self.keys[6*i+5], train=train)
            y = lnorm2(y)

        return y


if __name__ == "__main__":
    keys = random.split(random.key(7), 2)
    x = random.normal(keys[0], (1_024, 512), dtype=jnp.float32)
    enc = Encoder(key=keys[1])
    y = enc(x)
    print(x.shape)
    print(y.shape)

