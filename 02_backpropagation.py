"""
Backpropagation from scratch.

A two-layer network:   x --[w1]--> h --[w2]--> prediction

Small enough that every derivative can be computed by hand, which is the
point: the implementation is checked against those hand values rather than
being assumed correct.
"""
import numpy as np
import matplotlib.pyplot as plt
from fontTools.misc.bezierTools import epsilon

"""
returns the prediction and the intermediate parameter 
h(This is why framework like pytorch also remembers the value for the backward propagation)
, that required in backward propagation. consumes input(X), wright1(w1), weight2(w2).
    :param x: 
    :param w1: 
    :param w2: 
    :return: 
"""
def forward(x, w1, w2):
    h= x*w1
    prediction=h*w2
    return prediction, h


"""
:param cache:
:param prediction:
:param h
"""
def backward(cache, prediction, h):
    d_loss_d_pred = 2*(prediction-cache['y'])
    d_loss_d_w2 = d_loss_d_pred*h
    d_loss_d_h = d_loss_d_pred*cache['w2']
    d_loss_d_w1 = d_loss_d_h*cache['x']
    return d_loss_d_w2, d_loss_d_w1

def run_hand_calculated_network():
    """
        Check the implementation against derivatives computed by hand.

        Setup:  x = 2, w1 = 3, w2 = 4, target y = 30

        Forward:
            h    = w1 * x = 3 * 2 = 6
            pred = w2 * h = 4 * 6 = 24
            loss = (24 - 30)^2 = 36

        Backward, by hand:
            d(loss)/d(pred) = 2 * (24 - 30)     = -12
            d(loss)/d(w2)   = -12 * h  = -12*6  = -72
            d(loss)/d(h)    = -12 * w2 = -12*4  = -48
            d(loss)/d(w1)   = -48 * x  = -48*2  = -96
        """
    param_cache={
        'w1':3.0,
        'w2':4.0,
        'x':2.0,
        'y':30.0
    }
    prediction, h = forward(param_cache['x'], param_cache['w1'], param_cache['w2'])
    g_w2, g_w1 = backward(param_cache, prediction, h)
    expected_w1, expected_w2 = -96.0, -72.0
    print(f"forward: h = {h}, prediction = {prediction}, loss = {(prediction-param_cache['y']**2)}")
    print(f"gradient due to w1: {g_w1} .")
    print(f"gradient due to w2: {g_w2} .")
    assert np.isclose(expected_w1, g_w1), "w1 gradient does not match the derivation"
    assert np.isclose(expected_w2, g_w2), "w2 gradient does not match the derivation"
    print("both gradients match the hand computation")

def verify_numerically():
    """
    Check the gradient a second, independent way.

    The derivative is defined as: nudge the input slightly, see how much the
    output moves, divide by the nudge. If our backward pass is correct, it
    must agree with that measurement.

    This catches errors that a hand derivation might repeat -- if you
    differentiated wrongly on paper AND in code, the first check passes
    and this one fails.
    """

    x, y = 2.0, 30.0
    w1, w2 = 3.0, 4.0
    epsilon = 1e-5

    param_cache = {
        'w1': 3.0,
        'w2': 4.0,
        'x': 2.0,
        'y': 30.0
    }

    def loss_at(w1, w2):
        prediction, _= forward(x,w1, w2)
        return  (prediction-y)**2
    # verify numerically the change due to nudging both weights
    numeric_w1 = (loss_at(w1+epsilon, w2)- loss_at(w1-epsilon, w2))/(2*epsilon)
    numeric_w2 = (loss_at(w1, w2+epsilon)- loss_at(w1, w2-epsilon))/(2*epsilon)
    print(f"numeric_w1: {numeric_w1} \n numeric_w2: {numeric_w2}")

    prediction, h = forward(x, w1, w2)
    g_w2, g_w1 = backward(param_cache, prediction, h)

    assert np.isclose(numeric_w1, g_w1, rtol=1e-4), "there is error in calculated derivative using hand and code"
    assert np.isclose(numeric_w2, g_w2, rtol=1e-4), "there is error in calculated derivative using hand and code"


def experiment_vanishing_gradients():
    """
    The gradient reaching the first layer is a product of every layer's
    local derivative. Sigmoid's derivative is at most 0.25; ReLU's is 1.

    This is the arithmetic behind why deep networks were impractical
    before ReLU.
    """
    depths = range(1, 31)
    sigmoid_signal = [0.25 ** d for d in depths]
    relu_signal = [1.0 ** d for d in depths]

    plt.figure(figsize=(10, 6))
    plt.plot(depths, sigmoid_signal, marker='o', label='sigmoid (derivative <= 0.25)')
    plt.plot(depths, relu_signal, marker='s', label='ReLU (derivative = 1)')
    plt.yscale('log')
    plt.xlabel('number of layers')
    plt.ylabel('gradient reaching layer 1 (log scale)')
    plt.title('Why deep networks could not be trained before ReLU')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('experiments/vanishing_gradients.png', dpi=150, bbox_inches='tight')
    print("saved experiments/vanishing_gradients.png")


if __name__ == "__main__":
    run_hand_calculated_network()
    print()
    verify_numerically()
    print()
    experiment_vanishing_gradients()





