import jax.numpy as jnp
from jax import vmap
from jax import random

def softmax(x, axis):
    return jnp.exp(x) / jnp.expand_dims(jnp.sum(jnp.exp(x), axis), axis)

def attention_fn(query, key, value, mask: bool, dtype):
    d_k = query.shape[-1]
    attn_scores = jnp.dot(query, key.T) / jnp.sqrt(d_k)
    if mask:
        # print(attn_scores)
        attn_scores = attn_scores.at[
            jnp.triu_indices(attn_scores.shape[0], k=1)
        ].set(jnp.finfo(dtype).min)
        # print(attn_scores)
    attn = jnp.dot(softmax(attn_scores, 1), value)
    return attn

batched_attention_fn = vmap(attention_fn, (0, 0, 0, None, None), 0)

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
        self.dtype = dtype

    def __call__(self, input_embed, mask=False):
        query = jnp.dot(input_embed, self.weight_q)         # (n, d_k) @ (d_k, d_k) -> (n, d_k)
        key = jnp.dot(input_embed, self.weight_k)           # (n, d_k) @ (d_k, d_k) -> (n, d_k)
        value = jnp.dot(input_embed, self.weight_v)         # (n, d_k) @ (d_k, d_k) -> (n, d_k)
        output_embed = attention_fn(query, key, value,
                                    mask,
                                    self.dtype)             # -> (n, d_k)
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
        self.dtype = dtype

    def __call__(self, input_embed, mask=False):
        query = jnp.dot(input_embed, self.weight_q)         # (n, d_model) @ (h, d_model, d_k) -> (n, h, d_k)
        key = jnp.dot(input_embed, self.weight_k)           # (n, d_model) @ (h, d_model, d_k) -> (n, h, d_k)
        value = jnp.dot(input_embed, self.weight_v)         # (n, d_model) @ (h, d_model, d_v) -> (n, h, d_v)

        atten = batched_attention_fn(
            jnp.einsum("ijk->jik", query),                  # (n, h, d_k) -> (h, n, d_k)
            jnp.einsum("ijk->jik", key),                    # .
            jnp.einsum("ijk->jik", value),                  # .
            mask,
            self.dtype
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


# 2nd sublayer in decoder of transformer
class MultiHeadedCrossAttention:
    def __init__(self, d_model=512, num_heads=8, scale=1e-2, key=random.key(0), dtype=jnp.float32):
        self.d_model = d_model
        self.h = num_heads
        self.d_k = self.d_model // self.h
        self.d_v = self.d_k                                 # assuming d_v is same as d_k
        keys = random.split(key, 3)
        self.weight_q = scale * random.normal(keys[0], (self.h, self.d_model, self.d_k), dtype)
        self.weight_k = scale * random.normal(keys[1], (self.h, self.d_model, self.d_k), dtype)
        self.weight_v = scale * random.normal(keys[2], (self.h, self.d_model, self.d_v), dtype)
        self.dtype = dtype

    def __call__(self, input_embed, input_embed_query, mask=False):
        query = jnp.dot(input_embed_query, self.weight_q)   # (n, d_model) @ (h, d_model, d_k) -> (n, h, d_k)
        key = jnp.dot(input_embed, self.weight_k)           # (n, d_model) @ (h, d_model, d_k) -> (n, h, d_k)
        value = jnp.dot(input_embed, self.weight_v)         # (n, d_model) @ (h, d_model, d_v) -> (n, h, d_v)

        atten = batched_attention_fn(
            jnp.einsum("ijk->jik", query),                  # (n, h, d_k) -> (h, n, d_k)
            jnp.einsum("ijk->jik", key),                    # .
            jnp.einsum("ijk->jik", value),                  # .
            mask,
            self.dtype
        )

        atten = jnp.einsum("ijk->jik", atten)               # (h, n, d_k) -> (n, h, d_k)
        atten = atten.reshape(atten.shape[0], -1)           # (n, h, d_k) -> (n, h*d_k)

        return atten


if __name__ == "__main__":
    rand_keys = random.split(random.key(7), 4)
    dtype = jnp.float32
    num = 1_024
    d_model = 512
    num_heads = 8

    input_embed = random.normal(rand_keys[1], (num, d_model), dtype)
    shsa = SingleHeadedSelfAttention(d_model, key=rand_keys[0], dtype=dtype)
    out_embed_s = shsa(input_embed)
    print(out_embed_s.shape)

    mhsa = MultiHeadedSelfAttention(d_model, num_heads, key=rand_keys[2], dtype=dtype)
    out_embed_m = mhsa(input_embed)
    print(out_embed_m.shape)

    #####

    test_input_embed = random.normal(rand_keys[3], (3, d_model), dtype)
    masked_output_embed_s = shsa(test_input_embed, mask=True)
    masked_output_embed_s = mhsa(test_input_embed, mask=True)

