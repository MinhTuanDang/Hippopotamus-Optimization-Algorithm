# 🦛 Hippopotamus Optimization Algorithm (HOA) — Python Implementation

**Designed and Developed by Mohammad Hussien Amiri and Nastaran Mehrabi Hashjin (2024)** 

> “A novel nature-inspired metaheuristic optimization algorithm inspired by the complex social, territorial, and defensive behaviors of hippopotamuses.”  
> — *Scientific Reports*, 2024 ([DOI: 10.1038/s41598-024-54910-3](https://doi.org/10.1038/s41598-024-54910-3))

---

## 🧠 Overview

The **Hippopotamus Optimization Algorithm (HOA)** is a **nature-inspired metaheuristic algorithm** proposed by *Amiri & Mehrabi Hashjin (2024)*.  
It simulates the intelligent, social, and environmental behaviors of hippopotamuses in their natural habitats — particularly **river foraging, predator defense, and escape strategies**.

This repository contains a **fully engineered Python implementation** of HOA, featuring:
- ✅ Pure NumPy/Sklearn CPU version  
- ⚙️ Parallel multi-core version using `joblib`  
- 🚀 GPU-accelerated version using **CuPy (CUDA)**  
- 💾 Modular OOP design for research or industrial reuse  

---

## 📚 Algorithmic Background

The Hippopotamus Optimization Algorithm is based on **three behavioral phases** reflecting real-world actions of hippopotamuses:

| Phase | Behavior | Computational Analogy |
|--------|-----------|----------------------|
| **1. River/Pond Exploration** | Searching for new food and social areas | Exploration of search space, random group formation |
| **2. Defense Against Predators** | Coordinated defense maneuvers | Population diversity enhancement using Lévy flights |
| **3. Escape Strategy** | Evasive movement from predators | Exploitation phase for local search near promising solutions |

The algorithm dynamically switches between **exploration and exploitation** to maintain a balance between global search and convergence.

Mathematically, HOA integrates random group-based movement, dynamic exponential decay (`exp(-t/T)`), and **Lévy distribution**-based jumps for non-linear exploration.

---

## ⚙️ Features

### Core Features
- Full implementation of all three HOA phases.
- Direct correspondence to MATLAB code from the original paper.
- Object-Oriented design for reuse and scalability.
- IEEE-style documentation.

### Computational Enhancements
| Version | Technology | Description |
|----------|-------------|-------------|
| **CPU Base** | NumPy + Scikit-learn | Faithful port of MATLAB version with sklearn utilities (`MinMaxScaler`, `check_random_state`) |
| **Parallel CPU** | Joblib | Fitness evaluations distributed across all CPU cores |
| **GPU (HOA-GPU)** | CuPy | All matrix and random operations executed on CUDA-enabled GPUs |

### Additional Enhancements
- Auto-fallback from GPU → CPU if CUDA not detected.
- Deterministic execution with random state control.
- Vectorized distance computations via `pairwise_distances` (sklearn).
- Modular fitness evaluation layer for integration with ML models or simulations.

---

## 🧩 Installation

### Requirements
python >= 3.9
numpy >= 1.23
scikit-learn >= 1.3
joblib >= 1.3
cupy >= 13.0   # Optional, only for GPU acceleration

### Clone repository
git clone https://github.com/MinhTuanDang/Hippopotamus-Optimization-Algorithm/
cd Hippopotamus-Optimization-Algorithm

# (Optional) Create a clean virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

### Integration with ML Pipelines 🧪

The HOA optimizer can be used for:

Neural network hyperparameter tuning

Feature selection optimization

Engineering design problems

Combinatorial and continuous optimization

# Example with sklearn model tuning:
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from hippopotamus_optimization_parallel import HippopotamusOptimizer

X, y = load_iris(return_X_y=True)

def fitness(params):
    C, gamma = params
    model = SVC(C=abs(C), gamma=abs(gamma))
    return 1 - np.mean(cross_val_score(model, X, y, cv=5))

optimizer = HippopotamusOptimizer(30, 100, bounds=(0.001, 10), dim=2, fitness_func=fitness)
best_score, best_params, _ = optimizer.optimize()
print("Best SVM Params:", best_params)

### Theoretical Notes 🧬 

The core mechanism of HOA is driven by non-linear adaptation and stochastic dominance:

Uses dynamic exponential decay 
𝑇
=
𝑒
−
𝑡
/
𝑇
𝑚
𝑎
𝑥
T=e
−t/T
max
	​


Employs Lévy flight distribution for escape movement

Balances exploration and exploitation via time-adaptive switching

The algorithm’s stochastic operators are carefully tuned to ensure global convergence and robustness across multimodal landscapes.



