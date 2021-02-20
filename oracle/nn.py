import numpy as np
from numpy import random
from oracle import activations
from oracle import losses

class Net:
    def __init__(self, arch, loss="mse"):
        np.random.seed(1)

        weights = [np.random.rand(arch[i+1]["size"], arch[i]["size"]) * 0.0001
                for i in range(len(arch[1::]))]
        biases = [np.zeros((arch[i+1]["size"], 1)) 
                for i in range(len(arch[1::]))]

        self.activationFns = []
        for l in arch[1::]:
            if l["activation"] == "sigmoid":
                self.activationFns.append(activations.Sigmoid())
            if l["activation"] == "relu":
                self.activationFns.append(activations.Relu())

        if loss == "mse":
            self.lossFn = losses.MSE()
        if loss == "log":
            self.lossFn = losses.Logistic()

        self.biases = biases
        self.weights = weights

    def forward(self, x):
        actvs = [x]
        sums = []
        for w, b, activation in zip(self.weights, self.biases, self.activationFns):
            x = w.dot(x)+b
            sums.append(x)

            x = activation.calc(x)
            actvs.append(x)

        return sums, actvs

    def cost(self, a, y):
        m = a.shape[1]
        return 1/m * np.sum(self.lossFn.calc(a, y))

    def backprop(self, sums, actvs, y):
        m = actvs[-1].shape[1]

        a = actvs[-1]
        z = sums[-1]

        da = 1/m * self.lossFn.calc_deriv(a, y)
        dz = da * self.activationFns[-1].calc_deriv(z)

        dw = dz.dot(actvs[-2].T)
        db = dz

        deltas_w = [dw]
        deltas_b = [db]

        for w, z, a, activate in zip(self.weights[-1::-1], sums[-2::-1],
                actvs[-3::-1], self.activationFns[-2::-1]):
            dz = w.T.dot(dz) * activate.calc_deriv(z)
            dw = dz.dot(a.T)
            db = dz

            deltas_w.append(dw)
            deltas_b.append(db)
        return deltas_w[::-1], deltas_b[::-1]

    def update(self, m, deltas_w, deltas_b, lr):
        weights = []
        biases = []
        for w, b, dw, db in zip(self.weights, self.biases, deltas_w, deltas_b):
            w -= dw * lr
            b -= db * lr
            weights.append(w)
            biases.append(b)
        self.weights = weights
        self.biases = biases

    def train(self, x, y, epochs, lr):
        i = 0
        for i in range(epochs):
            actvs, sums = self.forward(x)
            dw, db = self.backprop(actvs, sums, y)
            m = actvs[0].shape[1]
            self.update(m, dw, db, lr)
            if i % 100 == 0:
                print(1/m * self.cost(actvs[-1], y))
            i += 1
