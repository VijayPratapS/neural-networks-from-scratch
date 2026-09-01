import matplotlib.pyplot as plt

"""
Gradient descent from scratch.

Minimizes f(x) = x^2, whose minimum is at x = 0.
We know the answer, which is the point: it lets us verify the algorithm
against ground truth, and study how the learning rate changes its behaviour.
"""

def f(x):
    return x**2

def gradient(x):
    return 2*x

def descend(steps, learning_rate, start):
    """
        Walk downhill from `start`, taking `steps` steps.

        Returns the full path, so we can plot what happened rather than
        only reporting where it ended up.
    """
    x= start
    path = [x]
    for _ in range(steps):
       x=x-learning_rate*gradient(x)
       path.append(x)
    return path

def experiment_learning_rates():
    """
    Four learning rates on the same problem, showing four distinct behaviours.

    This is the point of the file: the same algorithm, on the same function,
    from the same starting point, does completely different things depending
    on one number.
    """
    rates = [0.1, 0.5, 1.0, 1.05]
    labels = [
        "lr=0.1  — smooth convergence",
        "lr=0.5  — one exact jump to the minimum",
        "lr=1.0  — oscillates forever, never converges",
        "lr=1.05 — diverges"
    ]

    plt.figure(figsize=(10, 6))
    for rate, label in zip(rates, labels):
        path = descend(start=10.0, learning_rate=rate, steps=15)
        plt.plot(path, marker='o', label=label)

    plt.axhline(0, color='black', linestyle='--', linewidth=0.8, label='the minimum')
    plt.xlabel('step')
    plt.ylabel('x')
    plt.title('Gradient descent: the learning rate decides everything')
    plt.legend()
    plt.ylim(-30, 30)
    plt.savefig('experiments/learning_rates.png', dpi=150, bbox_inches='tight')
    print("saved experiments/learning_rates.png")


if __name__ == "__main__":
    experiment_learning_rates()