import numpy as np
from numpy import random
from oracle import actvs
from oracle import loss

class Net:
    def __init__(self, arch, cost):
        self.arch = arch 

        np.random.seed(1)

        weights = [np.random.rand(arch[i+1]["size"], arch[i]["size"]) * 0.0001
                for i in range(len(arch[1::]))]
        biases = [np.zeros((arch[i+1]["size"], 1)) 
                for i in range(len(arch[1::]))]

        self.actvs = []
        for l in arch[1::]:
            if l["activation"] == "sigmoid":
                self.actvs.append(actvs.Sigmoid())
            if l["activation"] == "relu":
                self.actvs.append(actvs.Relu())

        if cost == "mse":
            self.loss = loss.MSE()
        if cost == "log":
            self.loss = loss.Logistic()

        self.arch = arch
        self.biases = biases
        self.weights = weights

    def cost(self, a, y):
        m = a.shape[1]
        return 1/m * np.sum(self.loss.calc(a, y))

    def forward(self, x):
        actvs = [x]
        sums = []
        for w, b, actv in zip(self.weights, self.biases, self.actvs):
            x = w.dot(x)+b
            sums.append(x)

            x = actv.calc(x)
            actvs.append(x)

        return sums, actvs

    def backprop(self, sums, actvs, y):
        m = actvs[-1].shape[1]

        a = actvs[-1]
        z = sums[-1]

        dz = 1/m * self.loss.calc_deriv(a, y) * self.actvs[-1].calc_deriv(z)

        dw = dz.dot(actvs[-2].T)
        db = dz

        deltas_w = [dw]
        deltas_b = [db]

        for w, z, a, actv in zip(self.weights[-1::-1], sums[-2::-1],
                actvs[-3::-1], self.actvs[-2::-1]):
            dz = w.T.dot(dz) * actv.calc_deriv(z)
            dw = dz.dot(a.T)
            db = dz

            deltas_w = deltas_w + [dw]
            deltas_b = deltas_b + [db]
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
            sum = np.sum(self.loss.calc(actvs[-1], y))
            if i % 100 == 0:
                print(1/m * sum)
            i += 1
