import jax.numpy as jnp
from jax import vmap
from jax import random
from ffn import relu, FFN

def layer_norm(g, sigma, a, mu, b):
    return jnp.dot((g / sigma), (a - mu)) + b

batched_mean = vmap(jnp.mean, 0, 0)
batched_std = vmap(jnp.std, 0, 0)
batched_layer_norm = vmap(layer_norm, (None, 0, 0, 0, None), 0)

class Layernorm:
    def __init__(self, key=random.key(0), embed_shape=(512,), dtype=jnp.float32):
        keys = random.split(key, 3)
        self.ffn = FFN(keys[0], (embed_shape[-1],)*2, dtype=dtype)
        self.g = random.normal(keys[1], embed_shape, dtype=dtype)
        self.b = random.normal(keys[2], embed_shape, dtype=dtype)

    def __call__(self, x, activation=relu, batched=True):
        a = self.ffn(x, activation=activation, batched=batched)
        mu = batched_mean(a) if batched else jnp.mean(a)
        sigma = batched_std(a) if batched else jnp.std(a)
        # print(mu.shape, sigma.shape)
        h = batched_layer_norm(self.g, sigma, a, mu, self.b) if batched \
                else layer_norm(self.g, sigma, mu, a, self.b)
        h = activation(h)
        return h

if __name__ == "__main__":
    x = random.normal(random.key(0), (1_024, 512), dtype=jnp.float32)
    lnorm = Layernorm()
    y = lnorm(x, batched=True)
    print(x.shape)
    print(y.shape)
