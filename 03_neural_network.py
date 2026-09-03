"""
A neural network from scratch.

Architecture:  x --> [2 neurons with ReLU] --> output

Trained on a relationship that rises and then falls -- a shape no linear
model can represent. The second experiment in this file removes the ReLU
and shows the same network failing on the same data, which is the point:
the non-linearity is not decoration.
"""

import numpy as np
import matplotlib.pyplot as plt

def forward(params, use_relu=True, x=0.0):
    """
        Push one input through the network.

        Returns the prediction plus every intermediate value, because the
        backward pass needs them -- each local derivative depends on something
        computed on the way in.

        use_relu=False removes the activation, which is the ablation experiment.
    """

    w1, b1, w2, b2, p, q = params
    z1 = w1*x+b1
    z2 = w2*x+b2
    if use_relu:
        a = max(0.0, z1)
        b = max(0.0, z2)
    else:
        a, b= z1,z2
    prediction = ((a*p)+(b*q))
    return prediction, a, b, z1, z2

def backward(cache, params,  y, x, use_relu=True):
    pred , a, b, z1, z2 = cache
    w1, b1, w2, b2, p, q = params

    # impact on loss due to layer 2
    d_loss_d_pred = 2*(pred-y)
    d_loss_d_a = d_loss_d_pred*p
    d_loss_d_p = d_loss_d_pred*a
    d_loss_d_b = d_loss_d_pred*q
    d_loss_d_q = d_loss_d_pred*b

    # impact on loss due to layer 1 , propagated from layer 2
    d_loss_d_z1 = d_loss_d_a*((1.0 if z1>0 else 0.0) if use_relu else 1.0)
    d_loss_d_z2 = d_loss_d_b*((1.0 if z2>0 else 0.0)  if use_relu else 1.0)
    d_loss_d_w1 = d_loss_d_z1*x
    d_loss_d_w2 = d_loss_d_z2*x
    d_loss_d_b1 = d_loss_d_z1
    d_loss_d_b2 = d_loss_d_z2

    return np.array([d_loss_d_w1, d_loss_d_b1, d_loss_d_w2, d_loss_d_b2, d_loss_d_p, d_loss_d_q])
def train(X, Y, use_relu=True, lr= 0.01, epochs=3000, seed=3):
    """
        Gradient descent over the six parameters.

        Gradients are summed across all examples, then averaged -- because the
        loss is the MEAN squared error, so its gradient is a mean too. Without
        the division, the step size would depend on how many examples you have.
    """
    rng = np.random.default_rng(seed)
    params = rng.normal(size=6)*0.5
    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        grads=np.zeros(6)
        for x, y in zip(X,Y):
            cache = forward(params, use_relu, x)
            grads += backward(cache, params, y, x, use_relu)
            total_loss += (cache[0]-y)**2
        params=params-lr*grads/len(X)
        history.append(total_loss/len(X))
    return params, history

def experiment_with_and_without_relu():
    """
    The same network, the same data, the same training -- with and without
    the activation function.

    The data rises to a peak and then falls. Removing the ReLU makes the
    network linear:

        prediction = p(w1*x + b1) + q(w2*x + b2)
                   = (p*w1 + q*w2)*x + (p*b1 + q*b2)

    Six parameters collapse into a single straight line. It cannot represent
    a shape that goes up and then down, regardless of how long it trains.
    """
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    Y = np.array([1.0, 2.0, 1.0, 0.0, -1.0])

    params_relu, hist_relu = train(X, Y, use_relu=True)
    print("params [w1,b1,w2,b2,p,q]:", params_relu)
    for x in X:
        pred, a, b, z1, z2 = forward( params_relu, True, x)
        print(f"x={x}  z1={z1:+.3f}  z2={z2:+.3f}")
    params_lin, hist_lin = train(X, Y, use_relu=False)

    grid = np.linspace(0.5, 5.5, 100)
    pred_relu = [forward(params_relu, True, x)[0] for x in grid]
    pred_lin = [forward( params_lin, False, x)[0] for x in grid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(X, Y, s=80, zorder=3, label='data')
    ax1.plot(grid, pred_relu, label='with ReLU')
    ax1.plot(grid, pred_lin, linestyle='--', label='without ReLU')
    ax1.set_xlabel('input'); ax1.set_ylabel('output')
    ax1.set_title('The same network, with and without the activation')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(hist_relu, label='with ReLU')
    ax2.plot(hist_lin, linestyle='--', label='without ReLU')
    ax2.set_xlabel('epoch'); ax2.set_ylabel('mean squared error')
    ax2.set_title('Training loss')
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiments/why_relu.png', dpi=150, bbox_inches='tight')

    print(f"final loss with ReLU:    {hist_relu[-1]:.5f}")
    print(f"final loss without ReLU: {hist_lin[-1]:.5f}")
    print("predictions with ReLU:", [round(forward(params_relu, True, x)[0], 2) for x in X])
    print("targets:               ", list(Y))


def experiment_zero_initialization():
    """
    Every parameter starts at zero.

    Forward:  z1 = 0*x + 0 = 0, so a = 0. Same for b. prediction = 0*0 + 0*0 = 0.
    Backward: d_p = d_pred * a = d_pred * 0 = 0   (a is zero)
              d_a = d_pred * p = d_pred * 0 = 0   (p is zero)
              ... every gradient is a product containing a zero.

    Nothing updates, so the parameters stay at zero forever. This is why
    random initialization is required, not merely preferred.
    """
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    Y = np.array([1.0, 2.0, 1.0, 0.0, -1.0])

    params = np.zeros(6)
    for _ in range(1000):
        grads = np.zeros(6)
        for x, y in zip(X, Y):
            param_cache = forward(params, True, x)
            grads += backward(param_cache, params, y, x, True)
        params = params - 0.01 * grads / len(X)

    print(f"parameters after 1000 epochs of zero-initialized training: {params}")
    print("unchanged -- every gradient was zero, so nothing ever updated")


if __name__ == "__main__":
    experiment_with_and_without_relu()
    print()
    experiment_zero_initialization()







