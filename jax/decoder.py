import jax.numpy as jnp
from jax import random
from attention import MultiHeadedSelfAttention, MultiHeadedCrossAttention
from layernorm import Layernorm
from ffn import FFN
from dropout import dropout

class Decoder:
    def __init__(self, key=random.key(0), stack_size=6, dropout_p=0.1, dtype=jnp.float32):
        self.n = stack_size
        self.keys = random.split(key, 9*self.n)
        self.stack = [(
            MultiHeadedSelfAttention(key=self.keys[9*i], dtype=dtype),
            Layernorm(key=self.keys[9*i+1], dtype=dtype),
            MultiHeadedCrossAttention(key=self.keys[9*i+2], dtype=dtype),
            Layernorm(key=self.keys[9*i+3], dtype=dtype),
            FFN(key=self.keys[9*i+4], dtype=dtype),
            Layernorm(key=self.keys[9*i+5], dtype=dtype)
        ) for i in range(self.n)]
        self.dropout_p = dropout_p

    def __call__(self, x, y_enc, train=True):
        y = x
        for i, (m_mhsa, lnorm1, mhca, lnorm2, ffn, lnorm3) in enumerate(self.stack):
            y = y + dropout(m_mhsa(input_embed=x, mask=True), self.dropout_p,
                            key=self.keys[9*i+6], train=train)
            y = lnorm1(y)
            y = y + dropout(mhca(input_embed=y_enc, input_embed_query=y), self.dropout_p,
                            key=self.keys[9*i+7], train=train)
            y = lnorm2(y)
            y = y + dropout(ffn(y), self.dropout_p,
                            key=self.keys[9*i+8], train=train)
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

