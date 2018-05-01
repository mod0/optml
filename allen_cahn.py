import numpy as np
import functools as ft

def InitializeModel():
    #  Solves the Allen Cahn problem from 
    #  M.P. Calvo and A. Gerisch.
    #  Linearly implicit Runge-Kutta methods and approximate matrix 
    #  factorization. Applied Numerical Mathematics,
    #  53(2):183-200, 2005.

    # Parameters defining the RHS behavior.
    myalpha = 1
    mygamma = 10

    # Grid size in x and y
    M = 128
    N = 128

    # Suggested integration information.
    Tspan = np.array([0, 0.075])
    dt_integration = 0.002

    # linear representation of x and y variables
    x = np.linspace(0,1,M)
    y = np.linspace(0,1,N)

    # Grid representation of x and y.
    X, Y = np.meshgrid(x,y)

    # grid representation of initial condition.
    u0 = 0.4 + 0.1 * (X + Y) + 0.1 * (np.sin(10*X) * np.sin(20*Y))
    
    # grid representation of boundary values corresponding to the intitial condition.
    bound  = lambda i,j,dx,dy: (0.4 + 0.1 * ((j - 1) * dx + (i - 1) * dy) + 
                               0.1 * (np.sin(10 * (j - 1) * dx) * np.sin(20 * (i - 1) *dy)))

    # linear representation of initial condition.
    y0     = np.reshape(u0, M*N)
    
    # RHS and Jacobian function calls, in MATLAB standard form.
    rhsFun = ft.partial(AllenCahnNeumann, M=M, N=N, alpha=myalpha, gamma=mygamma, bound=bound)

    model  = {'rhsFun': rhsFun,
             'y0':     y0,
             'Tspan':  Tspan,
             'dt':     dt_integration,
             'M':      M,
             'N':      N,
             'alpha':  myalpha,
             'gamma':  mygamma}

    return model
             
    
def AllenCahnNeumann(u, M, N, alpha, gamma, bound):
    dx = 1.0/(N-1)
    dy = 1.0/(M-1)
    f  = np.zeros(M*N)
    
    for i in xrange(1, M+1):
        for j in xrange(1, N+1):        
            self = i + M * (j - 1) - 1 

            if ( i == 1 ):
                down = u[self] - dy * bound(i, j, dx, dy)
            else:
                down = u[i - 1 + M * (j - 1) - 1]
                
            if ( i == M ):
                up = u[self] + dy * bound(i,j,dx,dy)
            else:
                up = u[i + 1 + M * (j - 1) - 1]
        
            if ( j == 1 ):
                left = u[self] - dx * bound(i,j,dx,dy)
            else:
                left = u[i + M * (j - 2) - 1]
        
            if ( j == N ):
                right = u[self] + dx * bound(i,j,dx,dy)
            else:
                right = u[i + M * j - 1]
        
            uself = u[self]
            
            f[self] = alpha * (1.0/dy**2 * (up    - 2 * uself + down)  + \
                               1.0/dx**2 * (right - 2 * uself + left)) + gamma * (uself - uself**3)
            
    return f



