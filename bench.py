import timeit

import numpy as np

n, d = 1000, 2
rng = np.random.default_rng(42)
X = rng.normal(size=(n, d))
mu = np.mean(X, axis=0)
inv_cov = np.linalg.pinv(np.cov(X, rowvar=False))


def v1_loop():
    def single(x):
        delta = x - mu
        return np.sqrt(max(delta @ inv_cov @ delta, 0.0))

    return np.array([single(x) for x in X])


def v2_vectorized():
    diff = X - mu
    return np.sqrt(np.maximum(np.sum((diff @ inv_cov) * diff, axis=1), 0.0))


def v3_einsum():
    diff = X - mu
    return np.sqrt(
        np.maximum(np.einsum("ni,ij,nj->n", diff, inv_cov, diff, optimize=True), 0.0)
    )


t1 = timeit.timeit(v1_loop, number=1000)
t2 = timeit.timeit(v2_vectorized, number=1000)
t3 = timeit.timeit(v3_einsum, number=1000)

print(f"Loop Python:  {t1:.4f}s")
print(f"Vectorizat:   {t2:.4f}s  ({((t1 - t2) / t2) * 100:.1f}% mai slow loop-ul)")
print(f"Einsum:       {t3:.4f}s  ({((t1 - t3) / t3) * 100:.1f}% mai slow loop-ul)")
