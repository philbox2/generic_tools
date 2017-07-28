#!/usr/bin/python

"""explicit Euler and Runge-Kutta-4th order to solve ODEs numerically."""

import numpy as np

class integrate_euler_method:
    """Take the ordinary differential equation and integrate it using the explicit euler method
    
    Parameters:
    -----------
    ode: Function that defines the ordinary differential equation.
    y: current system state
    t_axis: array like points of time that the ode is solved at
    dt: stepwidth in time
  
    """

    def __init__(self, ff):
        self.ff = ff

    def set_initial_values(self, y0, t0):
        self.y = y0
        self.tt = t0

    def next_euler_method(self, t_n, y_n, h):
        y_next = y_n + h * self.ff(t_n, y_n)

        return y_next



    def integrate(self, tt, h):
        if self.y is None or self.tt is None:
            raise ValueError("Initialise class first with initial values.")
        
        y_current = self.next_euler_method( tt, self.y, h)
        self.y = y_current
        
        return y_current

    def integrate_along_taxis(self, t_axis, y0, show_progress=False):

        def print_progress(x):
            print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(x * 50), x * 100), end="", flush=True)
            
        def check_prerequisites_set_stepwidth():
            #check equally spaced:
            if np.diff(t_axis).std() >= 10**-6:
                raise ValueError("t_axis is not equally spaced")
                #Return stepwidth:

            return np.diff(t_axis).mean()

        hh = check_prerequisites_set_stepwidth()
        
        res_euler = []

        y_current = y0.copy()
        
        tN = len(t_axis)-1
        for idx, tt in enumerate(t_axis):
            if show_progress:
                print_progress(idx/tN)
            res_euler.append(y_current)
            y_current = self.next_runge_kutta_4th( tt, y_current, hh)

        return np.asarray(res_euler)


class integrate_runge_kutta_4thorder:
    """Find the stepwise solution of the ode f using the 4th order Runge Kutta method. Sadly the formulas are taken from internet resources aka wikipedia...so be carefull

    Parameters:
    -----------
    f: ordinary differential equation takes arguments f(t, y) = d/dt. Return value has to be an np.ndarray. Fix this ??
    y: system state at this time as np.ndarray
    ts: time_axis. That has to be checked to be equally spaced

    """

    def __init__(self, ff):
        self.ff = ff
        self.y = None
        self.tt = None

    def set_initial_values(self, y0, t0):
        self.y = y0
        self.tt = t0
        
    def next_runge_kutta_4th(self, t_n, y_n, h):
        """Find the next value of rk4"""
        k1 = self.ff( t_n, y_n )
        k2 = self.ff( t_n + .5*h, y_n + .5 * h * k1)
        k3 = self.ff( t_n + .5*h, y_n + .5 * h * k2)
        k4 = self.ff( t_n + h, y_n + h * k3)

        y_next = y_n + (1/6.) * h *( k1 + 2*k2 + 2*k3 + k4 )
        
        return y_next


        

    def integrate(self, tt, h):
        if self.y is None or self.tt is None:
            raise ValueError("Initialise class first with initial values.")
        
        y_current = self.next_runge_kutta_4th( tt, self.y, h)
        self.y = y_current
        
        return y_current

    def integrate_along_taxis(self, t_axis, y0, show_progress=False):

        def print_progress(x):
            print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(x * 50), x * 100), end="", flush=True)
            
        def check_prerequisites_set_stepwidth():
            #check equally spaced:
            if np.diff(t_axis).std() >= 10**-6:
                raise ValueError("t_axis is not equally spaced")
                #Return stepwidth:

            return np.diff(t_axis).mean()

        hh = check_prerequisites_set_stepwidth()
        
        res_rk4 = []

        y_current = y0.copy()
        
        tN = len(t_axis)-1
        for idx, tt in enumerate(t_axis):
            if show_progress:
                print_progress(idx/tN)
            res_rk4.append(y_current)
            y_current = self.next_runge_kutta_4th( tt, y_current, hh)
        if show_progress:
            print()
        return np.asarray(res_rk4)
