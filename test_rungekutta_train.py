from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from rungekutta45_train import rk45
from sklearn import linear_model

class SGDModel:
    def __init__(self):
        self.model = linear_model.SGDClassifier(warm_start=True)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class PerceptronModel:
    def __init__(self):
        self.model = linear_model.Perceptron(warm_start=True)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
if __name__ == "__main__":

    # # linear problem
    # odefun                    = lambda y: -y/2
    # options                   = {}
    # options['abstol']         = 1.0e-6
    # options['reltol']         = 1.0e-6 
    # options['numcheckpoints'] = 100
    # tspan                     = [0, 1.0]
    # y0                        = [2, 1.0]

    # oscillatory problem
    # odefun                    = lambda y: [-np.sin(y[2]), -100 * np.sin(100*y[2]), 1] 
    # options                   = {}
    # options['abstol']         = 1.0e-6
    # options['reltol']         = 1.0e-6 
    # options['numcheckpoints'] = 100
    # tspan                     = [0.0, 2 * math.pi]
    # y0                        = [1.0, 1.0, 0]
    
    # Allen Cahn
    import allen_cahn as ac
    model                     = ac.InitializeModel()
    odefun                    = lambda y: model['rhsFun'](y)
    options                   = {}
    options['abstol']         = 1.0e-8
    options['reltol']         = 1.0e-8 
    options['numcheckpoints'] = 1000
    options['nobservations']  = 10
    options['model']          = SGDModel()
    tspan                     = [0.0, 0.3]
    y0                        = model['y0']

    trajectory                = rk45(odefun, tspan, y0, options)
    
    # print number of accepted/rejected steps
    print("Accepted Steps:" + str(options['nacc']))
    print("Rejected Steps:" + str(options['nrej']))
    
    # Plot trajectory
    n = len(y0)

    # create a figure
    plt.figure()
    plt.hold(True)
    
    for i in xrange(2):
        plt.plot(trajectory[i, :])

    plt.hold(False)
    plt.show()
