import numpy as np

class Sigmoid:
    def calc(self, z):
        return 1/(1+np.exp(-z))
    def calc_deriv(self, z):
        z = self.calc(z)
        return z*(1-z)

class Relu:
    def calc(self, z):
        return z * (z > 0)
    def calc_deriv(self, z):
        return 1. * (z > 0)
