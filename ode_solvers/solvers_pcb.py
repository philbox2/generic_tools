#!/usr/bin/python

"""explicit Euler and Runge-Kutta-4th order to solve ODEs numerically."""

import numpy as np

def integrate_euler_method(f, y, t_axis, dt):
    """Take the ordinary differential equation and integrate it using the explicit euler method
    
    Parameters:
    -----------
    ode: Function that defines the ordinary differential equation.
    y: current system state
    t_axis: array like points of time that the ode is solved at
    dt: stepwidth in time
  
    """

    def next_euler_method(t_n, y_n, h):
        y_next = y_n + h * f(t_n, y_n)

        return y_n

    def check_prerequisites_set_stepwidth():
        #check equally spaced:
        if np.diff(t_axis).std() >= 10**-6:
            raise ValueError("t_axis is not equally spaced")
        #Return stepwidth:

        return np.diff(t_axis).mean()

    h = check_prerequisites_set_stepwidth()

    res_euler = []
    y_current = y.copy()

    for idx, tt in enumerate(t_axis):
        
        res_euler.append(y_current)
        y_current = next_euler_method( tt, y_current, h)
        
    return np.asarray(res_euler)


def integrate_runge_kutta_4thorder(f, t_axis, y,  h=None):
    """Find the stepwise solution of the ode f using the 4th order Runge Kutta method. Sadly the formulas are taken from internet resources aka wikipedia...so be carefull

    Parameters:
    -----------
    f: ordinary differential equation takes arguments f(t, y) = d/dt. Return value has to be an np.ndarray. Fix this ??
    y: system state at this time
    ts: time_axis. That has to be checked to be equally spaced

    """
    def next_runge_kutta_4th(t_n, y_n, h):
        """Find the next value of rk4"""
        k1 = f( t_n, y_n )
        k2 = f( t_n + .5*h, y_n + .5 * h * k1)
        k3 = f( t_n + .5*h, y_n + .5 * h * k2)
        k4 = f( t_n + h, y_n + h * k3)

        y_next = y_n + (1/6.) * h *( k1 + 2*k2 + 2*k3 + k4 )
        #print( y_next)
        return y_next


    
    def check_prerequisites_set_stepwidth():
        #check equally spaced:
        if np.diff(t_axis).std() >= 10**-6:
            raise ValueError("t_axis is not equally spaced")
        
        #Return stepwidth:
        return np.diff(t_axis).mean()

    h = check_prerequisites_set_stepwidth()


    res_rk4 = []
    y_current = y.copy()

    for idx, tt in enumerate(t_axis):
        
        res_rk4.append(y_current)
        y_current = next_runge_kutta_4th( tt, y_current, h)
        
    return np.asarray(res_rk4)
