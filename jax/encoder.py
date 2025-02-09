import jax.numpy as jnp
from jax import random
from attention import MultiHeadedSelfAttention
from ffn import FFN
from layernorm import Layernorm

class Encoder:
    def __init__(self, key=random.key(0), stack_size=6, dtype=jnp.float32):
        self.n = stack_size
        keys = random.split(key, 4*self.n)
        self.stacks = [(
            MultiHeadedSelfAttention(key=keys[4*i], dtype=dtype),
            Layernorm(key=keys[4*i+1], dtype=dtype),
            FFN(key=keys[4*i+2], dtype=dtype),
            Layernorm(key=keys[4*i+3], dtype=dtype)
        ) for i in range(self.n)]

    def __call__(self, x):
        y = x
        for (mhsa, lnorm1, ffn, lnorm2) in self.stacks:
            y = y + mhsa(y)
            y = lnorm1(y)
            y = y + ffn(y)
            y = lnorm2(y)

        return y


if __name__ == "__main__":
    keys = random.split(random.key(7), 2)
    x = random.normal(keys[0], (1_024, 512), dtype=jnp.float32)
    enc = Encoder(key=keys[1])
    y = enc(x)
    print(x.shape)
    print(y.shape)

