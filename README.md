# Neural Networks from Scratch

Implementations of gradient descent, backpropagation, and a working neural network, built in NumPy from the mathematics — no ML frameworks.

The point of this repository is not to reproduce what PyTorch already does well. It is to derive each component by hand, implement it, and then **run the experiments that show what breaks when a component is removed or misconfigured.** The failure modes are the interesting part.

---

## Contents

| File | What it contains |
|---|---|
| `01_gradient_descent.py` | Gradient descent from scratch, and the four distinct behaviours of the learning rate |
| `02_backpropagation.py` | The chain rule applied layer by layer, verified against hand-computed derivatives |
| `03_neural_network.py` | A small network that learns a non-linear function without any framework |
| `experiments/` | Plots produced by the files above |

---

## 1. Gradient descent, and what the learning rate actually decides

Minimizing `f(x) = x²`, whose minimum is at `x = 0`. The answer is known in advance, which is the point — it lets the algorithm be checked against ground truth.

The update rule is one line:

```python
x = x - learning_rate * gradient(x)
```

Everything else in the file is measurement.

![learning rates](experiments/learning_rates.png)

Four learning rates, same function, same starting point `x = 10`, four qualitatively different outcomes:

| learning rate | behaviour | why |
|---|---|---|
| 0.1 | smooth convergence | steps are small enough that each one improves on the last |
| **0.5** | **reaches the minimum in a single step** | for `f(x) = x²`, `x − 0.5·(2x) = 0` for **any** starting `x` |
| **1.0** | **oscillates between +10 and −10 forever** | the step is exactly wide enough to land on the mirror image; the loss never changes |
| 1.05 | diverges | each bounce overshoots slightly further than the last |

The two middle rows were derived by hand before the code was written, then confirmed by running it.

**`lr = 0.5` is exact, not lucky:**

```
x − 0.5 · 2x = x − x = 0        for any x
```

For this particular function, 0.5 is the optimal step — the analogue of what second-order methods (Newton's method) try to compute in general.

**`lr = 1.0` is the knife edge:**

```
x = 10  →  10 − 1.0 · 20 = −10
x = −10 →  −10 − 1.0 · (−20) = +10
```

Neither converging nor diverging — a permanent standoff at unchanging loss. In real training this appears as a loss curve that flattens and stops improving, which is why learning-rate schedules decay the step size over time rather than holding it fixed.

---

## 2. Backpropagation

The chain rule applied backward through a network: each layer receives the gradient from the layer ahead of it, multiplies by its own local derivative, and passes the result back.

Worked by hand on a two-layer network first (`x → w₁ → h → w₂ → prediction`), then implemented:

```
d(loss)/d(w₂) = d(loss)/d(pred) × h
d(loss)/d(w₁) = d(loss)/d(pred) × w₂ × x
```

The implementation's gradients are verified against these hand-computed values.

**The vanishing gradient, made concrete.** Because each layer contributes a multiplicative factor, the gradient reaching the first layer is a product of every layer's local derivative:

```
sigmoid, derivative ≤ 0.25:   0.25²⁰ ≈ 0.0000000000009   →  the first layer receives nothing
ReLU, derivative = 1:         1²⁰ = 1                     →  the gradient arrives intact
```

This is the arithmetic behind why deep networks were impractical before ReLU, and it is reproduced in `experiments/`.

---

## 3. A network that learns a non-linear function

A two-neuron network with ReLU, trained to fit a relationship that rises and then falls — a shape no linear model can represent.

**Why the non-linearity is not optional.** Without an activation function, stacked layers collapse:

```
prediction = w₂(w₁x + b₁) + b₂ = (w₂w₁)x + (w₂b₁ + b₂)
```

Two layers, six parameters, and the result is still a single straight line. Depth buys nothing. The experiment in `experiments/why_relu.py` shows the same network failing on the same data with the activation removed.

**Where the bend comes from.** ReLU always switches at zero — of *its own input*, which is `w·x + b`. So the bend appears at `x = −b/w`, and the bias is what slides it. Different neurons learn different biases, place their bends in different positions, and their combination builds a curve out of straight segments.

**Zero initialization fails completely.** With every parameter at zero, every activation is zero, so every gradient is zero (each gradient is a product containing a zero factor), so nothing ever updates. The network is frozen at initialization. This is why random initialization is required, not merely preferred.

---

## Running it

```bash
pip install -r requirements.txt
python 01_gradient_descent.py
```

Each file writes its plots to `experiments/`.

---

## Notes on approach

Every component here was derived on paper before being written in code, and the hand-computed values are used as the test: if the implementation and the derivation disagree, one of them is wrong.

The mathematical background this rests on — linear algebra, calculus, probability, and statistics, worked through from first principles — is in a separate repository: [ml-research-foundations](https://github.com/VijayPratapS/ml-research-foundations).
