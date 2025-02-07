import jax.numpy as jnp
from jax import vmap
from jax import random

def softmax(x, axis):
    return jnp.exp(x) / jnp.expand_dims(jnp.sum(jnp.exp(x), axis), axis)

def attention_fn(query, key, value):
    d_k = query.shape[-1]
    return jnp.dot(softmax(jnp.dot(query, key.T) / jnp.sqrt(d_k), 1), value)

batched_attention_fn = vmap(attention_fn, (0, 0, 0), 0)

def linear_params(m, n, key, scale=1, dtype=jnp.float32):
    t = jnp.sqrt(1/m)
    w_key, b_key = random.split(key)
    w = scale * random.uniform(w_key, (n, m), minval=-t, maxval=t, dtype=dtype)
    b = scale * random.uniform(b_key, (n,), minval=-t, maxval=t, dtype=dtype)
    return w, b


class SingleHeadedSelfAttention:
    def __init__(self, d_model=512, scale=1e-2, key=random.key(0), dtype=jnp.float32):
        self.d_k = d_model
        keys = random.split(key, 3)
        self.weight_q = scale * random.normal(keys[0], (self.d_k, self.d_k), dtype)
        self.weight_k = scale * random.normal(keys[1], (self.d_k, self.d_k), dtype)
        self.weight_v = scale * random.normal(keys[2], (self.d_k, self.d_k), dtype)

    def __call__(self, input_embed):
        query = jnp.dot(input_embed, self.weight_q)         # (n, d_k) @ (d_k, d_k) -> (n, d_k)
        key = jnp.dot(input_embed, self.weight_k)           # (n, d_k) @ (d_k, d_k) -> (n, d_k)
        value = jnp.dot(input_embed, self.weight_v)         # (n, d_k) @ (d_k, d_k) -> (n, d_k)
        output_embed = attention_fn(query, key, value)      # -> (n, d_k)
        return output_embed


class MultiHeadedSelfAttention:
    def __init__(self, d_model=512, num_heads=8, scale=1e-2, key=random.key(0), dtype=jnp.float32):
        self.d_model = d_model
        self.h = num_heads
        self.d_k = self.d_model // self.h
        self.d_v = self.d_k                                 # assuming d_v is same as d_k
        keys = random.split(key, 3)
        self.weight_q = scale * random.normal(keys[0], (self.h, self.d_model, self.d_k), dtype)
        self.weight_k = scale * random.normal(keys[1], (self.h, self.d_model, self.d_k), dtype)
        self.weight_v = scale * random.normal(keys[2], (self.h, self.d_model, self.d_v), dtype)

    def __call__(self, input_embed):
        query = jnp.dot(input_embed, self.weight_q)         # (n, d_model) @ (h, d_model, d_k) -> (n, h, d_k)
        key = jnp.dot(input_embed, self.weight_k)           # (n, d_model) @ (h, d_model, d_k) -> (n, h, d_k)
        value = jnp.dot(input_embed, self.weight_v)         # (n, d_model) @ (h, d_model, d_v) -> (n, h, d_v)

        atten = batched_attention_fn(
            jnp.einsum("ijk->jik", query),                  # (n, h, d_k) -> (h, n, d_k)
            jnp.einsum("ijk->jik", key),                    # .
            jnp.einsum("ijk->jik", value),                  # .
        )

        """ # rawdogging # ;) worked
        query = jnp.einsum("ijk->jik", query)               # (n, h, d_k) -> (h, n, d_k)
        key = jnp.einsum("ijk->jki", key)                   # (n, h, d_k) -> (h, d_k, n)
        value = jnp.einsum("ijk->jik", value)               # (n, h, d_k) -> (h, n, d_k)

        resp = jnp.matmul(query, key) / jnp.sqrt(self.d_k)  # (h, n, d_k) @ (h, d_k, n) -> (h, n, n)
        resp_softm = softmax(resp, 2)                       # ... -> ...
        atten_1 = jnp.matmul(resp_softm, value)             # (h, n, n) @ (h, n, d_k) -> (h, n, d_k)
        print(jnp.allclose(atten_1, atten).sum())
        """

        atten = jnp.einsum("ijk->jik", atten)               # (h, n, d_k) -> (n, h, d_k)
        atten = atten.reshape(atten.shape[0], -1)           # (n, h, d_k) -> (n, h*d_k)

        return atten


if __name__ == "__main__":
    rand_key = random.key(7)
    dtype = jnp.float32
    num = 1_024
    d_model = 512
    num_heads = 8

    shsa = SingleHeadedSelfAttention(d_model, key=rand_key, dtype=dtype)
    input_embed = random.normal(rand_key, (num, d_model), dtype)
    out_embed_s = shsa(input_embed)
    print(out_embed_s.shape)

    mhsa = MultiHeadedSelfAttention(d_model, num_heads, key=rand_key, dtype=dtype)
    out_embed_m = mhsa(input_embed)
    print(out_embed_m.shape)

