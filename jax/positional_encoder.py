import jax.numpy as jnp
import matplotlib.pyplot as plt

class Sinusoidal:
    def __init__(self, n, d_model=512, factor=10_000, dtype=jnp.float32):
        self.d_model = d_model
        self.n = n
        self.f = factor
        self.dtype = dtype

    def __call__(self):
        pos_enc = jnp.broadcast_to(
            jnp.arange(self.n, dtype=jnp.float32)[:, None],
            shape=(self.n, self.d_model)
        )                                                                                                   # (n, d_model)
        # print(pos_enc)

        for j in range(self.d_model):
            if j % 2 == 0:
                pos_enc = pos_enc.at[:, j].set(jnp.sin(pos_enc[:, j] * jnp.pow(self.f, -j/self.d_model)))
            else:
                pos_enc = pos_enc.at[:, j].set(jnp.cos(pos_enc[:, j] * jnp.pow(self.f, -(j-1)/self.d_model)))

        # print(pos_enc)

        return pos_enc


if __name__ == "__main__":
    n = 500
    pos_encoder = Sinusoidal(n=n, d_model=4)
    pos_enc = pos_encoder()
    # print(pos_enc.shape)

    plt.scatter(jnp.arange(n), pos_enc[:, -2], s=0.2)
    plt.scatter(jnp.arange(n), pos_enc[:, -1], s=0.2)
    plt.show()

