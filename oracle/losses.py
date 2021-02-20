import numpy as np

class MSE:
    def calc(self, a, y):
        return (a-y)**2

    def calc_deriv(self, a, y):
        return 2*(a-y)

class Logistic:
    def calc(self, a, y):
        return -(y*np.log(a)+(1-y)*np.log(1-a))
    def calc_deriv(self, a, y):
        return (y-a)/(a*(a-1))
