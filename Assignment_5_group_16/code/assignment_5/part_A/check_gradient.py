import numpy as np
from NN import Convolution2D, Dense, MaxPooling2D, Softmax, CrossEntropy

np.random.seed(7)
EPS = 1e-5


def numerical_gradient(array, objective):
    grad = np.zeros_like(array, dtype=np.float64)

    for idx in np.ndindex(array.shape):
        old_value = array[idx]

        array[idx] = old_value + EPS
        plus = objective()

        array[idx] = old_value - EPS
        minus = objective()

        array[idx] = old_value
        grad[idx] = (plus - minus) / (2 * EPS)

    return grad


def relative_error(analytical, numerical):
    numerator = np.linalg.norm(analytical - numerical)
    denominator = (
        np.linalg.norm(analytical)
        + np.linalg.norm(numerical)
        + 1e-12
    )
    return numerator / denominator


# 1. Dense layer: compare dW against finite differences.
dense = Dense(3, 2)
x = np.array([0.2, -0.4, 0.7], dtype=np.float64)
upstream = np.array([0.3, -0.6], dtype=np.float64)

def dense_objective():
    return np.sum(dense.forward(x) * upstream)

dense.forward(x)
dense.backward(upstream)
dense_numeric = numerical_gradient(dense.weights, dense_objective)

print("Dense weight gradient error:",
      relative_error(dense.dW, dense_numeric))


# 2. Convolution: compare dW and db against finite differences.
conv = Convolution2D(
    input_channels=1,
    output_channels=2,
    kernel_size_height=3,
    kernel_size_width=3,
    stride=1,
    padding=0,
)

x_conv = np.random.randn(1, 5, 5)
conv.forward(x_conv)
upstream_conv = np.random.randn(*conv.Z.shape)

def conv_objective():
    return np.sum(conv.forward(x_conv) * upstream_conv)

conv.forward(x_conv)
conv.backward(upstream_conv)

conv_numeric_w = numerical_gradient(conv.weights, conv_objective)
conv_numeric_b = numerical_gradient(conv.biases, conv_objective)

print("Conv weight gradient error:",
      relative_error(conv.dW, conv_numeric_w))
print("Conv bias gradient error:",
      relative_error(conv.db, conv_numeric_b))


# 3. Max pooling: use distinct input values to avoid max ties.
pool = MaxPooling2D(pool_size=2, stride=2)
x_pool = np.array([[
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 9.0, 7.0, 8.0],
    [2.0, 3.0, 6.0, 7.0],
    [4.0, 5.0, 8.0, 10.0],
]])

pool.forward(x_pool)
upstream_pool = np.random.randn(*pool.Z.shape)

def pool_objective():
    return np.sum(pool.forward(x_pool) * upstream_pool)

pool.forward(x_pool)
pool_analytic = pool.backward(upstream_pool)
pool_numeric = numerical_gradient(x_pool, pool_objective)

print("MaxPool input gradient error:",
      relative_error(pool_analytic, pool_numeric))

print("\nExpected: gradient errors close to zero.")



softmax = Softmax()
loss_fn = CrossEntropy()

logits = np.array([0.4, -0.2, 0.7, 0.1, -0.5])
target = np.array([0., 0., 1., 0., 0.])

def objective():
    probs = softmax.forward(logits)
    return loss_fn.forward(target, probs)

probs = softmax.forward(logits)
loss = loss_fn.forward(target, probs)

grad_loss = loss_fn.backward()
grad_logits = softmax.backward(grad_loss)

numeric_logits = numerical_gradient(logits, objective)

print("Loss:", loss)
print("Analytical gradient:", grad_logits)
print("Numerical gradient:", numeric_logits)
print("Gradient error:", relative_error(grad_logits, numeric_logits))

assert np.allclose(grad_logits, probs - target, atol=1e-7)