import numpy as np
from oracle import nn

def test_backprop():
    # TODO: Maybe randomize the net each time?
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
    ])

    losses = ["log", "mse"]

    # TODO: Maybe we should randomize data each time?
    x = np.array([1, 10, 1, 0.2]).reshape(-1, 1)
    y = np.array([0.4])

    for loss in losses:
        net.loss = loss
        assert grad_check(net, x, y) < 1e-9

# flatten_params flattens given weights and biases into one array.
def flatten_params(weights, biases):
    params = []
    for w in weights:
        w = w.flatten()
        params += [x for x in w]
    for b in biases:
        b = b.flatten()
        params += [x for x in b]
    return params

# erect_params transforms given parameters to shaped weights and biases.
def erect_params(params, weights, biases):
    # erect slices a specific portion of params and then shapes it to
    # a desired paramater, namely, param.
    def erect(params, param, at):
        cut = param.shape[0] * param.shape[1]
        param_cut = np.array(params[at:at+cut]).reshape(param.shape)
        at += cut

        return param_cut, at

    # this is used to track the current position of params
    z = 0 
    
    erect_weights = []
    erect_biases = []

    for w in weights:
        param, z = erect(params,w, z)
        erect_weights.append(param)

    for b in biases:
        param, z = erect(params, b, z)
        erect_biases.append(param)

    return erect_weights, erect_biases

# calc_cost calculates cost when nudging i'th parametere by some eps.
def calc_cost(params, i, net, eps, x, y):
    nudged_params = params.copy()
    nudged_params[i] = nudged_params[i] + eps

    w_orig, b_orig = net.weights, net.biases

    w, b = erect_params(nudged_params, net.weights, net.biases) 

    # apply nudged paramters to the net and calculate the cost
    net.weights = w
    net.biases = b
    sums, actv = net.forward(x)
    cost = net.cost(actv[-1], y)

    # revert net's parameters
    net.weights = w_orig
    net.biases = b_orig

    return cost

# performs gradient checking and returns the sucess value. Anything below 
# 1e-4 is considered to be bad. 
def grad_check(net, x, y, eps=1e-6):
    params = flatten_params(net.weights, net.biases)
    derivs = []

    for i in range(len(params)):
        # approximate the derivitive: (J(theta+eps)-J(theta))/2eps
        plus = calc_cost(params, i, net, eps, x, y)
        minus = calc_cost(params, i, net, -eps, x, y)

        dp = (plus-minus)/ (2*eps)
        derivs.append(dp)

    # compute derivitives with backpropagation
    sums, actv = net.forward(x) 
    dw, db = net.backprop(sums, actv, y)

    backprop_params = flatten_params(dw, db)

    # because we appended to an array, we have to convert it to a numpy
    # array for future comparisons. However, we might be able to append to a 
    # numpy array in the approximation step, then we wouldn't need this 
    numerical_params = np.array(derivs)

    numer = np.linalg.norm(numerical_params - backprop_params)
    denom = np.linalg.norm(numerical_params) + np.linalg.norm(backprop_params)

    return numer / denom
