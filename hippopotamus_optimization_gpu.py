"""
hippopotamus_optimization_gpu.py
--------------------------------
GPU-Accelerated Hippopotamus Optimization Algorithm (HOA-GPU)
Author: Code GPT
Reference: Amiri & Mehrabi Hashjin (2024)
DOI: 10.1038/s41598-024-54910-3
"""

import numpy as np
from sklearn.utils import check_random_state
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed

# Try GPU mode
try:
    import cupy as cp
    GPU_AVAILABLE = True
    xp = cp  # unified namespace (xp = numpy or cupy)
    print("🚀 GPU mode enabled via CuPy")
except ImportError:
    GPU_AVAILABLE = False
    xp = np
    print("⚙️ CuPy not found, using CPU (NumPy fallback)")

class HippopotamusOptimizerGPU:
    """
    GPU-accelerated Hippopotamus Optimization Algorithm (HOA-GPU)

    Parameters
    ----------
    n_agents : int
        Number of agents
    n_iterations : int
        Number of iterations
    bounds : tuple(float, float)
        Search range
    dim : int
        Dimensionality
    fitness_func : callable
        Objective function to minimize
    use_gpu : bool
        If False, fallback to CPU automatically
    random_state : int
        Reproducible RNG seed
    """

    def __init__(self, n_agents, n_iterations, bounds, dim, fitness_func,
                 use_gpu=True, random_state=None, scale_inputs=False):
        self.n_agents = n_agents
        self.n_iterations = n_iterations
        self.lower_bound, self.upper_bound = bounds
        self.dim = dim
        self.fitness_func = fitness_func
        self.scale_inputs = scale_inputs
        self.random_state = check_random_state(random_state)
        self.use_gpu = use_gpu and GPU_AVAILABLE

        self.xp = cp if self.use_gpu else np

        # Initialize population
        X = self.lower_bound + self.random_state.rand(n_agents, dim) * (self.upper_bound - self.lower_bound)
        if self.scale_inputs:
            scaler = MinMaxScaler(feature_range=(self.lower_bound, self.upper_bound))
            X = scaler.fit_transform(X)

        self.X = self.xp.asarray(X)
        self.fitness = self._evaluate(self.X)

    def _evaluate(self, X):
        """Evaluate fitness (GPU-aware)"""
        if self.use_gpu:
            X_host = cp.asnumpy(X)
        else:
            X_host = X
        results = Parallel(n_jobs=-1)(delayed(self.fitness_func)(x) for x in X_host)
        return self.xp.asarray(results)

    def levy_flight(self, n_agents, dim, beta=1.5):
        """GPU Lévy flight"""
        xp = self.xp
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = xp.random.normal(0, sigma_u, (n_agents, dim))
        v = xp.random.normal(0, 1, (n_agents, dim))
        return 0.05 * u / (xp.abs(v) ** (1 / beta))

    def _clip(self, X):
        return xp.clip(X, self.lower_bound, self.upper_bound)

    def optimize(self, verbose=True):
        xp = self.xp
        best_score = float(xp.min(self.fitness))
        best_pos = xp.copy(self.X[xp.argmin(self.fitness)])
        curve = []

        for t in range(1, self.n_iterations + 1):
            half = self.n_agents // 2
            RL = self.levy_flight(self.n_agents, self.dim)
            Dominant = xp.copy(best_pos)
            T = xp.exp(-t / self.n_iterations)

            # === PHASE 1 === (Exploration)
            rand_vals = xp.random.rand(half, self.dim)
            group_ids = xp.random.randint(0, self.n_agents, size=half)
            mean_groups = self.X[group_ids]

            X_P1 = self.X[:half] + rand_vals[:, :1] * (Dominant - self.X[:half])
            if T > 0.6:
                X_P2 = self.X[:half] + rand_vals * (Dominant - mean_groups)
            else:
                mask = xp.random.rand(half) > 0.5
                X_P2 = xp.empty_like(X_P1)
                X_P2[mask] = self.X[:half][mask] + rand_vals[mask] * (mean_groups[mask] - Dominant)
                X_P2[~mask] = self.lower_bound + xp.random.rand(xp.sum(~mask), self.dim) * (self.upper_bound - self.lower_bound)

            X_P1, X_P2 = self._clip(X_P1), self._clip(X_P2)
            F_P1 = self._evaluate(X_P1)
            F_P2 = self._evaluate(X_P2)

            improved = F_P1 < self.fitness[:half]
            self.X[:half][improved], self.fitness[:half][improved] = X_P1[improved], F_P1[improved]
            improved = F_P2 < self.fitness[:half]
            self.X[:half][improved], self.fitness[:half][improved] = X_P2[improved], F_P2[improved]

            # === PHASE 2 === (Defense)
            predators = self.lower_bound + xp.random.rand(half, self.dim) * (self.upper_bound - self.lower_bound)
            F_HL = self._evaluate(predators)

            dist = xp.linalg.norm(predators[:, None, :] - self.X[half:], axis=2)
            b = xp.random.uniform(2, 4, size=half)
            c = xp.random.uniform(1, 1.5, size=half)
            d = xp.random.uniform(2, 3, size=half)
            l = xp.random.uniform(-2 * np.pi, 2 * np.pi, size=half)

            X_P3 = xp.zeros_like(self.X[half:])
            for i in range(half):
                denom = (c[i] - d[i] * xp.cos(l[i]))
                if self.fitness[half + i] > F_HL[i]:
                    X_P3[i] = RL[half + i] * predators[i] + (b[i] / denom) * (1 / (dist[i] + 1e-9))
                else:
                    X_P3[i] = RL[half + i] * predators[i] + (b[i] / denom) * (1 / (2 * dist[i] + xp.random.rand(self.dim)))
            X_P3 = self._clip(X_P3)
            F_P3 = self._evaluate(X_P3)

            improved = F_P3 < self.fitness[half:]
            self.X[half:][improved], self.fitness[half:][improved] = X_P3[improved], F_P3[improved]

            # === PHASE 3 === (Exploitation)
            LO_LOCAL, HI_LOCAL = self.lower_bound / t, self.upper_bound / t
            D_opts = [2 * xp.random.rand(self.dim) - 1, xp.random.rand(1), xp.random.randn()]
            D = D_opts[int(xp.random.randint(0, 3))]
            X_P4 = self.X + xp.random.rand(self.n_agents, 1) * (LO_LOCAL + D * (HI_LOCAL - LO_LOCAL))
            X_P4 = self._clip(X_P4)
            F_P4 = self._evaluate(X_P4)

            improved = F_P4 < self.fitness
            self.X[improved], self.fitness[improved] = X_P4[improved], F_P4[improved]

            # Update best
            idx_best = int(xp.argmin(self.fitness))
            if self.fitness[idx_best] < best_score:
                best_score = float(self.fitness[idx_best])
                best_pos = xp.copy(self.X[idx_best])

            curve.append(best_score)
            if verbose:
                print(f"[Iter {t:03d}/{self.n_iterations}] Best = {best_score:.6e}")

        if self.use_gpu:
            best_pos = cp.asnumpy(best_pos)
            curve = np.asarray(curve)
        return best_score, best_pos, curve


# Example
if __name__ == "__main__":
    def sphere(x):
        return float(np.sum(x ** 2))

    ho = HippopotamusOptimizerGPU(
        n_agents=64,
        n_iterations=200,
        bounds=(-5.12, 5.12),
        dim=30,
        fitness_func=sphere,
        use_gpu=True,
        random_state=42
    )
    best, pos, curve = ho.optimize()
    print("\n✅ Final Best:", best)
