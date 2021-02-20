import numpy as np
from oracle import nn

def flatten_net(net):
    params = []
    for w in net.weights:
        w = w.flatten()
        params += [x for x in w]
    for b in net.biases:
        b = b.flatten()
        params += [x for x in b]
    return params

# TODO: rename
def fit_net(params, net):
    z = 0
    weights_fit = []
    biases_fit = []
    for w in net.weights:
        cut = w.shape[0]*w.shape[1]
        ww = np.array(params[z:z+cut])
        z += cut
        weights_fit.append(ww.reshape(w.shape))
    for b in net.biases:
        cut = b.shape[0]*b.shape[1]
        bb = np.array(params[z:z+cut])
        z += cut
        biases_fit.append(bb.reshape(b.shape))

    return weights_fit, biases_fit

def calc_cost(params, i, net, eps, x, y):
    params1 = params.copy()
    params1[i] = params1[i] + eps

    w_orig, b_orig = net.weights, net.biases

    w, b = fit_net(params1, net) 
    net.weights = w
    net.biases = b

    sums, actv = net.forward(x)
    cost = net.cost(actv[-1], y)

    net.weights = w_orig
    net.biases = b_orig

    print(actv[-1], cost)
    return cost

def grad_check(net, x, y):
    params = flatten_net(net)
    derivs = []
    eps = 1e-6
    for i in range(len(params)):
        plus = calc_cost(params, i, net, eps, x, y)
        minus = calc_cost(params, i, net, -eps, x, y)

        dp = (plus-minus)/ (2*eps)
        derivs.append(dp)

    sums, actv = net.forward(x) 
    dw, db = net.backprop(sums, actv, y)
    # TODO: Weird flex, but ok
    net.weights = dw
    net.biases = db

    non_numerical = flatten_net(net)
    numerical = np.array(derivs)

    goodness = np.linalg.norm(numerical-non_numerical)
    godness_bottom = np.linalg.norm(numerical)+np.linalg.norm(non_numerical)

    return goodness/godness_bottom

def test_nn():
    net = nn.Net([
        {
            "size": 4
        },
        {
            "size": 5,
            "activation": "relu"
        },
        {
            "size": 3,
            "activation": "relu"
        },
        {
            "size": 4,
            "activation": "sigmoid"
        },
        {
            "size": 1,
            "activation": "sigmoid"
        }
    ], "log")

    x = np.array([1, 10, 1, 0.2]).reshape(-1, 1)
    y = np.array([0.4])
    assert grad_check(net, x, y) < 1e-9
