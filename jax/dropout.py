import jax.numpy as jnp
from jax import random, jit, lax

@jit
def dropout(x, dropout_p, key=random.key(0), train=True):
    retention_p = 1 - dropout_p

    def train_dropout(x):
        r = random.bernoulli(key=key, p=retention_p, shape=x.shape)
        return r * x

    def test_dropout(x):
        return retention_p * x

    y = lax.cond(train, train_dropout, test_dropout, x)
    return y


if __name__ == "__main__":
    dropout_p = 0.5
    keys = random.split(random.key(100), 2)

    x = random.normal(key=keys[0], shape=(10,), dtype=jnp.float32)

    y_train = dropout(x, dropout_p, key=keys[1])
    y_test = dropout(x, dropout_p, train=False)

    print(y_train)
    print(y_test)

