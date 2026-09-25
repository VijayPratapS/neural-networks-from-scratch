"""
How depth kills the gradient, and how residual connections rescue it.

We build networks with 1 to 20 hidden layers and measure ONE thing: how much
of the gradient survives the journey back to the first layer.

The gradient starts at 1, so whatever arrives IS the surviving fraction --
no division needed at the end.

Everything here uses ONE example (a 1-D input vector), which is why the
forward pass is `W @ x` rather than the batch form `X @ W.T`. One example is
easier to trace by hand, and the mechanism is identical.

Biases are omitted deliberately: d(z)/d(input) = W, so the bias disappears
when differentiating with respect to the input and has no effect on gradient
flow, which is all this file measures.
"""

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# BUILDING THE NETWORK
# ----------------------------------------------------------------------------

def build_network(depth, width=2, seed=1):
    """
    Create one weight matrix per layer.

    Each matrix is (width x width): `width` rows (one per neuron) and `width`
    columns (one per input). Keeping every layer the same size is REQUIRED for
    residual connections -- `input + layer(input)` is an addition, and you can
    only add things of the same length. This is exactly why the transformer
    paper fixes d_model = 512 throughout.

    Returns a list: [W0, W1, W2, ...], one matrix per layer.
    """
    rng = np.random.default_rng(seed)          # seeded -> the same weights every run
    layers = []
    for _ in range(depth):                     # `_` because we don't need the counter
        W = rng.normal(size=(width, width)) * 0.5
        layers.append(W)
    return layers


# ----------------------------------------------------------------------------
# FORWARD PASS
# ----------------------------------------------------------------------------

def forward(x, layers, use_residual=False):
    """
    Push x through every layer, recording what each one produced.

    Per layer:
        z = W @ input          the weights
        a = max(0, z)          ReLU
        out = input + a        the residual (only if use_residual=True)

    We must SAVE every intermediate value, because the backward pass needs to
    know what each layer received and what it produced.

    Returns the list of activations:
        [x, output_of_layer_0, output_of_layer_1, ...]
    """
    activations = [x]                          # starts with just the input

    for W in layers:                           # loop over the matrices themselves
        current_input = activations[-1]        # [-1] = the LAST item = the previous
                                               #        layer's output. This IS depth:
                                               #        each layer reads what came before.

        z = W @ current_input                  # each ROW of W is one neuron's weights
        a = np.maximum(0, z)                   # ReLU, applied to every entry at once

        if use_residual:
            a = current_input + a              # THE RESIDUAL: add the input back in

        activations.append(a)                  # record it for the backward pass

    return activations


# ----------------------------------------------------------------------------
# BACKWARD PASS
# ----------------------------------------------------------------------------

def backward(activations, layers, use_residual=False):
    """
    Send a gradient backwards and record its size at every layer.

    We start with a gradient of all 1s. We are not training -- we only want to
    know how much SURVIVES, and starting at 1 makes the answer readable directly.

    Per layer, walking backwards:
        gate  = 1 where z was positive, 0 where negative   (ReLU's derivative)
        block = grad * gate                                (element by element)
        back  = W.T @ block                                (through the weights)
        residual:  grad = grad + back        (keep your value AND add)
        no residual: grad = back             (replaced entirely -- can shrink to 0)

    Returns a list of gradient sizes, ordered from layer 0 to the output.
    """
    grad = np.ones_like(activations[-1])       # array of 1s, same shape as the output
    sizes = [np.abs(grad).mean()]              # abs() drops signs (we want SIZE, not
                                               # direction); mean() gives one number

    # walk BACKWARDS: last layer first
    for i in reversed(range(len(layers))):

        # recompute what this layer produced BEFORE ReLU -- we need it for the gate.
        # activations[i] is what layer i RECEIVED (activations[i+1] is what it produced).
        z = layers[i] @ activations[i]

        # ReLU's derivative: 1 where the gate was open, 0 where it was closed
        relu_gate = (z > 0).astype(float)      # True/False -> 1.0/0.0

        # `*` is element by element: each neuron's own gate blocks its own gradient
        blocked = grad * relu_gate

        # `@` with .T pushes the gradient back through the weights.
        # The transpose regroups the weights by INPUT instead of by NEURON, so each
        # input collects blame from ALL the neurons it fed. That summing is the point.
        through_layer = layers[i].T @ blocked

        if use_residual:
            # the bypass: the gradient keeps its own value AND adds what came
            # through the layer. It can never shrink below what it already was.
            grad = grad + through_layer
        else:
            # no bypass: the gradient is REPLACED by whatever survived the layer
            grad = through_layer

        sizes.append(np.abs(grad).mean())

    return sizes[::-1]                         # [::-1] reverses: we recorded from the
                                               # output backwards, but we want layer 0 first


# ----------------------------------------------------------------------------
# MEASUREMENT
# ----------------------------------------------------------------------------

def gradient_at_first_layer(depth, use_residual, width=2, seed=1):
    """Build a network of the given depth and return the gradient reaching layer 0."""
    layers = build_network(depth, width, seed)
    x = np.ones(width)
    activations = forward(x, layers, use_residual)
    sizes = backward(activations, layers, use_residual)
    return sizes[0]                            # sizes[0] is layer 0, after reversing


def check_against_hand_computation():
    """
    Verify the implementation against the two-layer example computed by hand.

    W0 = [[0.5, 0.5],        W1 = [[0.5, -0.5],       x = [1, 1]
          [0.5, 0.5]]              [0.5,  0.5]]

    Expected:  without residual -> [0.5, 0.5, 1.0]
               with residual    -> [3.0, 1.5, 1.0]
    """
    layers = [np.array([[0.5, 0.5], [0.5, 0.5]]),
              np.array([[0.5, -0.5], [0.5, 0.5]])]
    x = np.array([1.0, 1.0])

    acts = forward(x, layers, use_residual=False)
    plain = backward(acts, layers, use_residual=False)

    acts = forward(x, layers, use_residual=True)   # recompute: the residual changes
    residual = backward(acts, layers, use_residual=True)  # the forward pass too

    print("hand-computed check")
    print(f"  without residual: {plain}      expected [0.5, 0.5, 1.0]")
    print(f"  with residual:    {residual}   expected [3.0, 1.5, 1.0]")

    assert np.allclose(plain, [0.5, 0.5, 1.0]), "plain case does not match the derivation"
    assert np.allclose(residual, [3.0, 1.5, 1.0]), "residual case does not match"
    print("  both match\n")


def experiment_depth_vs_gradient():
    """Sweep depths 1 to 20 and plot the gradient reaching layer 0."""
    depths = range(1, 21)
    plain = [gradient_at_first_layer(d, use_residual=False) for d in depths]
    residual = [gradient_at_first_layer(d, use_residual=True) for d in depths]

    plt.figure(figsize=(10, 6))
    plt.plot(depths, plain, marker='o', label='plain layers')
    plt.plot(depths, residual, marker='s', label='with residual connections')
    plt.yscale('log')                          # without a log axis, everything below
                                               # depth 5 is a flat line against zero
    plt.xlabel('number of layers')
    plt.ylabel('gradient reaching layer 1 (log scale)')
    plt.title('Residual connections keep the gradient alive with depth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('experiments/residual_connections.png', dpi=150, bbox_inches='tight')
    print("saved experiments/residual_connections.png\n")

    print("depth      plain          residual        ratio")
    for d in [1, 5, 10, 15, 20]:
        p = gradient_at_first_layer(d, False)
        r = gradient_at_first_layer(d, True)
        print(f"{d:5d}   {p:12.10f}   {r:12.4f}   {r/p:>15,.0f}x")


if __name__ == "__main__":
    check_against_hand_computation()
    experiment_depth_vs_gradient()