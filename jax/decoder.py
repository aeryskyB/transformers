import jax.numpy as jnp
from jax import random
from attention import MultiHeadedSelfAttention, MultiHeadedCrossAttention
from layernorm import Layernorm
from ffn import FFN

class Decoder:
    def __init__(self, key=random.key(0), stack_size=6, dtype=jnp.float32):
        self.n = stack_size
        keys = random.split(key, 6*self.n)
        self.stack = [(
            MultiHeadedSelfAttention(key=keys[6*i], dtype=dtype),
            Layernorm(key=keys[6*i+1], dtype=dtype),
            MultiHeadedCrossAttention(key=keys[6*i+2], dtype=dtype),
            Layernorm(key=keys[6*i+3], dtype=dtype),
            FFN(key=keys[6*i+4], dtype=dtype),
            Layernorm(key=keys[6*i+5], dtype=dtype)
        ) for i in range(self.n)]

    def __call__(self, x, y_enc):
        y = x
        for (m_mhsa, lnorm1, mhca, lnorm2, ffn, lnorm3) in self.stack:
            y = y + m_mhsa(input_embed=x, mask=True)
            y = lnorm1(y)
            y = y + mhca(input_embed=y_enc, input_embed_query=y)
            y = lnorm2(y)
            y = y + ffn(y)
            y = lnorm3(y)
        return y


if __name__ == "__main__":
    key = random.key(7)
    keys = random.split(key, 3)
    x = random.normal(keys[0], (1_024, 512))
    y_enc = random.normal(keys[1], (1024, 512))
    dec = Decoder(key=keys[2])
    y = dec(x, y_enc)
    print(x.shape)
    print(y_enc.shape)
    print(y.shape)

