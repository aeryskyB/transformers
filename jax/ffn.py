import jax.numpy as jnp
from jax import random, vmap

def linear_params(m, n, key=random.key(0), scale=1, dtype=jnp.float32):
    t = jnp.sqrt(1/m)
    w_key, b_key = random.split(key)
    w = scale * random.uniform(w_key, (n, m), minval=-t, maxval=t, dtype=dtype)
    b = scale * random.uniform(b_key, (n,), minval=-t, maxval=t, dtype=dtype)
    return w, b

def relu(tensor):
    return jnp.maximum(0, tensor)

def linear_transform(x, w, b):
    return w @ x + b

batched_linear_transform = vmap(linear_transform, (0, None, None), 0)

class FFN:
    def __init__(self, key=random.key(0), layers=[512, 2048, 512], dtype=jnp.float32):
        keys = random.split(key, len(layers)-1)
        self.layers = [linear_params(m, n, key=keys[i], dtype=dtype) for i, (m, n) in enumerate(zip(layers[:-1], layers[1:]))]

    def __call__(self, x, activation=relu, batched=True):
        y = x
        for i, (w, b) in enumerate(self.layers):
            y = batched_linear_transform(y, w, b) if batched else linear_transform(y, w, b)
            if i != len(self.layers)-1:
                y = activation(y)
        return y

if __name__ == "__main__":
    # t = jnp.arange(-4, 5)
    # print(t)
    # print(relu(t))
    ffn = FFN()
    x = random.normal(key=random.key(0), shape=(1000, 512))
    y = ffn(x)
    print(y.shape)
