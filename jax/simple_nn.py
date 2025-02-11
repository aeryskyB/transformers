import sys
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp
from torchvision import datasets
from torchvision.transforms import ToTensor

def linear_params(m, n, key, scale=1, dtype=jnp.float32):
    t = jnp.sqrt(1/m)
    w_key, b_key = random.split(key)
    w = scale * random.uniform(w_key, (n, m), minval=-t, maxval=t, dtype=dtype)
    b = scale * random.uniform(b_key, (n,), minval=-t, maxval=t, dtype=dtype)
    return w, b

def random_linear_params(m, n, key, scale=1e-2, dtype=jnp.float32):
    w_key, b_key = random.split(key)
    w = scale * random.normal(w_key, (n, m), dtype=dtype)
    b = scale * random.uniform(b_key, (n,), dtype=dtype)
    return w, b

def init_nn(sizes, key):
    keys = random.split(key, len(sizes))
    return [linear_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def relu(x):
    return jnp.maximum(x, 0)

def predict(params, image):
    activations = image
    for w, b in params[:-1]:
        activations = relu(jnp.dot(w, activations) + b)
    fw, fb = params[-1]
    logits = jnp.dot(fw, activations) + fb
    return logits - logsumexp(logits)

batched_predict = vmap(predict, in_axes=(None, 0))

def one_hot(x, k, dtype=jnp.float32):
    return jnp.array(x[:, None] == jnp.arange(k)[None, :], dtype)

def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)

def loss(params, images, targets):
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)

@jit
def update(params, x, y, lr):
    grads = grad(loss)(params, x, y)
    return [(w - lr*dw, b - lr*db) for (w, b), (dw, db) in zip(params, grads)]

mnist_data_files = [
    "t10k-images-idx3-ubyte",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte",
    "t10k-labels-idx1-ubyte.gz",
    "train-images-idx3-ubyte",
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte",
    "train-labels-idx1-ubyte.gz",
]


if __name__ == "__main__":
    layer_sizes = [28*28, 1024, 128, 10]
    lr = 2e-2
    num_epoch = 10
    batch_size = 64
    num_targets = 10
    params = init_nn(layer_sizes, random.key(0))

    try:
        train = datasets.MNIST("./data", train=True, download=False, transform=ToTensor())
        test = datasets.MNIST("./data", train=False, download=False, transform=ToTensor())
    except RuntimeError:
        print("\033[38;2;255;0;0mMNIST data not downloaded. Downloading...\033[38;2;255;255;255m",
              file=sys.stderr)
        train = datasets.MNIST("./data", train=True, download=True, transform=ToTensor())
        test = datasets.MNIST("./data", train=False, download=True, transform=ToTensor())
        print("\033[38;2;0;255;0mDone!\033[38;2;255;255;255m", file=sys.stderr)

    train_data, train_labels = train.data.numpy(), train.targets.numpy()
    test_data, test_labels = test.data.numpy(), test.targets.numpy()
    num_pixels = train_data.shape[-2] * train_data.shape[-1]

    train_data = train_data.reshape(-1, num_pixels)
    test_data = test_data.reshape(-1, num_pixels)
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)

    for ep in range(num_epoch):
        for i in range(0, len(train_data), batch_size):
            x = train_data[i*batch_size:(i+1)*batch_size]
            y = train_labels[i*batch_size:(i+1)*batch_size]
            params = update(params, x, y, lr)

        train_acc = accuracy(params, train_data, train_labels)
        test_acc = accuracy(params, test_data, test_labels)
        print(f"epoch {ep:>2} | {train_acc=} | {test_acc=}")

