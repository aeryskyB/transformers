from jax import random

def dropout(x, dropout_p, key=random.key(0), train=True):
    assert dropout_p > 0 and dropout_p < 1

    retention_p = 1 - dropout_p

    if train:
        r = random.bernoulli(key=key, p=retention_p, shape=x.shape)
        y = r * x
    else:
        y = retention_p * x

    return y

