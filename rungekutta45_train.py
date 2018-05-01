from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import random
import traceback

class RollingAverage:
    def __init__(self, n):
        self.n = n
        self.head = 0
        self.window = np.zeros(n)

    def push(self, val):
        self.window[self.head] = val
        self.head += 1
        self.head %= self.n

    def mean(self):
        return np.mean(self.window)

    
def rk45(odefun, tspan, yini, options):
    """
    Runge-Kutta-Fehlberg 4(5) implementation for autonomous systems
    +------------++------------+------------+------------+------------+------------+------------+
    |     0      ||            |            |            |            |            |            |
    +------------++------------+------------+------------+------------+------------+------------+
    |    1/4     ||    1/4     |            |            |            |            |            |
    +------------++------------+------------+------------+------------+------------+------------+
    |    3/8     ||    3/32    |    9/32    |            |            |            |            |
    +------------++------------+------------+------------+------------+------------+------------+
    |   12/13    || 1932/2197  | -7200/2197 | 7296/2197  |            |            |            |
    +------------++------------+------------+------------+------------+------------+------------+
    |     1      ||  439/216   |     -8     |  3680/513  | -845/4104  |            |            |
    +------------++------------+------------+------------+------------+------------+------------+
    |    1/2     ||   -8/27    |     2      | -3544/2565 | 1859/4104  |   -11/40   |            |
    +------------++------------+------------+------------+------------+------------+------------+
    +------------++------------+------------+------------+------------+------------+------------+
    |    O(5)    ||   16/135   |     0      | 6656/12825 |28561/56430 |   -9/50    |    2/55    |
    +------------++------------+------------+------------+------------+------------+------------+
    |    O(4)    ||   25/216   |     0      | 1408/2565  | 2197/4104  |    -1/5    |     0      |
    +------------++------------+------------+------------+------------+------------+------------+
    """

    try:
        c       = np.zeros(6)
        c[:]    = [      0      ,     1/4    ,     3/8    ,    12/13   ,     1      ,     1/2    ]

        a       = np.zeros((6, 6))
        a[:,:]  = [[     0      ,      0     ,      0     ,      0     ,      0     ,      0     ],
                   [    1/4     ,      0     ,      0     ,      0     ,      0     ,      0     ],
                   [    3/32    ,    9/32    ,      0     ,      0     ,      0     ,      0     ],
                   [ 1932/2197  , -7200/2197 , 7296/2197  ,      0     ,      0     ,      0     ],
                   [  439/216   ,     -8     ,  3680/513  , -845/4104  ,      0     ,      0     ],
                   [   -8/27    ,     2      , 3544/2565 , 1859/4104  ,   -11/40   ,      0     ]]


        b       = np.zeros((2, 6))
        b[:,:]  = [[   16/135   ,     0      , 6656/12825 ,28561/56430 ,   -9/50    ,    2/55    ],
                   [   25/216   ,     0      , 1408/2565  , 2197/4104  ,    -1/5    ,     0      ]]

        # store the number of variables
        n       = len(yini)
        nstages = 6

        # check the existence of abstol and reltol
        if 'abstol' in options:
            abstol = options['abstol']
        else:
            abstol = 1.0e-4

        if 'reltol' in options:
            reltol = options['reltol']
        else:
            reltol = 1.0e-4

        # check for the existence of count of checkpoints
        if 'numcheckpoints' in options:
            numcheckpoints = options['numcheckpoints']
        else:
            numcheckpoints = 10

        # reset values of nacc and nrej
        options['nacc'] = 0
        options['nrej'] = 0

        # now divide up the interval with numcheckpoints
        timepoints = np.linspace(tspan[0], tspan[1], numcheckpoints + 1)

        print timepoints

        # create solution trajectory
        trajectory = np.zeros((n, numcheckpoints + 1))

        # store the initial solution
        trajectory[:, 0] = yini

        # stage values
        Y = np.zeros((n, nstages))

        # f values
        K = np.zeros((n, nstages))

        # store yini and compute fini
        Y[:, 0] = yini
        K[:, 0] = odefun(Y[:, 0])

        # Get h value from options if it exists
        if 'h' in options:
            h = options['h']
        else:
            h = 1.0e-3

        # get the machine epsilon
        eps = np.spacing(1)

        # rounding
        roundoff = eps/2.0
            
        # get the initial time
        t   = tspan[0]

        # get the final time
        tf  = tspan[1]

        # index for comparison against timepoints
        # to store trajectory when it exceeds this
        # or equals it
        i   = 0

        # safety factor
        fac = 0.8

        # number of variables to observe
        if 'nobservations' in options:
            nobs = options['nobservations']
        else:
            nobs = 10
            
        # create training mask by randomly selecting
        # variables to observe
        mask = random.sample(xrange(n), min(n, nobs))

        # Get hold of the model
        # expect a function model.train(X, y)
        # and model.predict(X)
        if 'model' in options:
            model = options['model']
        else:
            raise "Need an online model with train and predict methods."

        # Get hold of validation window size
        # window and success threshold
        if 'nvalidation' in options:
            nvalidation = options['nvalidation']
        else:
            nvalidation = 10

        # create a validation window
        # push one whenever prediction turned out to be
        # true
        validation_window = RollingAverage(nvalidation)

        # only if the validation window has atleast
        # threshold number of successes will the
        # model be used to predict.
        if 'valthreshold' in options:
            valthreshold = options['valthreshold']
        else:
            valthreshold = 0.60

        # a constant by which to shrink stepsize if next step
        # is predicted to be rejected.
        if 'shrinkconstant' in options:
            shrinkconstant = options['shrinkconstant']
        else:
            shrinkconstant = 0.8

        # maxtimes to shrink stepsize if next step
        # is predicted to be rejected.
        if 'maxshrinks' in options:
            maxshrinks = options['maxshrinks']
        else:
            maxshrinks = 5


        # Store nsamples window of past features and data
        if 'nsamples' in options:
            nsamples = options['nsamples']
        else:
            nsamples = 10

        # Create feature variable with nsamples
        # and additional feature for step size
        featureX    = np.zeros((1, len(mask) + 1))

        # For accept signal
        signalYAcc  = np.zeros(nsamples)
        featureXAcc = np.zeros((nsamples, len(mask) + 1))
        countAcc    = 0
        indexAcc    = 0

        # For reject signal
        signalYRej  = np.zeros(nsamples)
        featureXRej = np.zeros((nsamples, len(mask) + 1))
        countRej    = 0
        indexRej    = 0

        # Minimum number of training samples
        if 'minsamples' in options:
            minsamples = options['minsamples']
        else:
            minsamples = 10
            
        # past prediction
        acceptnext = None
        
        while tf - t - roundoff >= 0:
            if tf - t <= 10 * roundoff * abs(tf):
                break
            
            for s in xrange(nstages - 1):
                Y[:, s + 1] = Y[:, 0] + h * K.dot(a[s + 1, :])
                K[:, s + 1] = odefun(Y[:, s + 1])

            # final output and error
            yfin = Y[:, 0] + h * K.dot(b[0, :])
            yerr = h * K.dot((b[0, :] - b[1, :]))
            
            # find the scaling factor            
            sc   = abstol + np.maximum(np.abs(Y[:, 0]), np.abs(yfin)) * reltol

            # compute the error
            err  = max(np.sqrt(np.sum((yerr / sc)**2)/n), 1.0e-10)

            # accept or reject
            if err <= 1:
                # accept
                if acceptnext == True:
                    validation_window.push(1)
                else:
                    validation_window.push(0)
                    
                # train
                # append accept signal
                featureXAcc[indexAcc, :-1] = Y[mask, 0]
                featureXAcc[indexAcc, -1]  = h
                signalYAcc[indexAcc]       = 1            # accept signal
                countAcc                  += 1
                indexAcc                  += 1
                countAcc                   = min(nsamples, countAcc)
                indexAcc                  %= nsamples     # replace oldest pair

                # Only train if min num of samples are met
                # and atleast one training signal of each class
                if countAcc > 0 and countRej > 0 and (countAcc + countRej) > minsamples:
                    fX = np.vstack((featureXAcc[:countAcc,:], featureXRej[:countRej, :]))
                    sY = np.hstack((signalYAcc[:countAcc], signalYRej[:countRej]))
                    if __debug__:
                        print "Training Model"                   
                    model.train(fX, sY)
                
                # now store the final output as next step ini
                Y[:, 0] = yfin
                K[:, 0] = odefun(Y[:, 0])
            
                # store the solution
                if t + h - timepoints[i + 1] >= 0 or tf - t - h <= 10 * roundoff * abs(tf):
                    i                = i + 1
                    trajectory[:, i] = yfin

                # increment time by h
                t = t + h

                # increment the count of accepted steps
                options['nacc'] += 1

                # compute the new h for next timestep
                h = h * min(1.5, max(0.2, fac * (1/err)**(1/5)))

                # Only predict if you have already trained
                if countAcc > 0 and countRej > 0 and (countAcc + countRej) > minsamples:
                    # construct new feature with new h and Y value
                    featureX[0,:-1] = Y[mask, 0]
                    featureX[0, -1] = h
                    if __debug__:
                        print "Validating Model"                   
                    # prediction and validation
                    sY         = model.predict(featureX)
                    acceptnext = (sY == 1)

                # model is somewhat trustworthy
                # if next step won't be accepted,
                # shrink stepsize by shrink constant
                # test for step acceptance
                if validation_window.mean() > valthreshold:
                    if __debug__:
                        print "Using Model for Prediction"
                    if not acceptnext:
                        for j in xrange(maxshrinks):
                            h *= shrinkconstant
                            featureX[0, -1] = h
                            acceptnext = (model.predict(featureX) == 1)

                            if acceptnext:
                                break                    
            else:
                # reject
                if acceptnext == False:
                    validation_window.push(1)
                else:
                    validation_window.push(0)
                    
                # train
                # append reject signal
                featureXRej[indexRej, :-1] = Y[mask, 0]
                featureXRej[indexRej, -1]  = h
                signalYRej[indexRej]       = 0            # reject signal
                countRej                  += 1
                indexRej                  += 1
                countRej                   = min(nsamples, countRej)
                indexRej                  %= nsamples     # replace oldest pair

                # Only train if min num of samples are met
                # and atleast one training signal of each class
                if countAcc > 0 and countRej > 0 and (countAcc + countRej) > minsamples:
                    fX = np.vstack((featureXAcc[:countAcc,:], featureXRej[:countRej, :]))
                    sY = np.hstack((signalYAcc[:countAcc], signalYRej[:countRej]))
                    if __debug__:
                        print "Training Model"                  
                    model.train(fX, sY)

                # increment the count of rejected steps
                options['nrej'] += 1
               
                # compute the new h for next timestep
                h = h * min(1.0, max(0.2, fac * (1/err)**(1/5)))

                if countAcc > 0 and countRej > 0 and (countAcc + countRej) > minsamples:
                    # construct new feature with new h 
                    featureX[0, -1]  = h
                    if __debug__:
                        print "Validating Model"
                    # prediction and validation
                    sY         = model.predict(featureX)
                    acceptnext = (sY == 1)
                
                # model is somewhat trustworthy
                # if next step won't be accepted,
                # shrink stepsize by shrink constant
                if validation_window.mean() > valthreshold:
                    if __debug__:
                        print "Using Model for Prediction"
                    if not acceptnext:
                        for j in xrange(maxshrinks):
                            h *= shrinkconstant
                            featureX[0, -1] = h
                            acceptnext = (model.predict(featureX) == 1)

                            if acceptnext:
                                break                    

        return trajectory[:, :i]
    except BaseException as e:
        print traceback.print_exc()    
