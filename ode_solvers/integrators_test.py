#!/usr/bin/python3

"""
This script test the ode_solvers defined in the same directory with two
simple examples.

Just call this using python3 ingerators_test.py in shell.
"""
import numpy as np
import solvers_pcb

import imp
imp.reload(solvers_pcb)

from matplotlib import pyplot as plt

class test_solver:

    def __init__(self):
        self.h = 0.05
        self.t_axis = np.arange(0, 5, self.h)
        
        self.gg = 9.81
        self.beta = 1
        self.mm = 85
        
        
    def ode_decay(self, t, y):
        """RHS of exponetial decay"""

        dy_dt = y

        return dy_dt
    
    def integrate_ode_decay(self):
       
        #Sovle the decay ode for y_0 = 1:
        y_decay = np.array([1])
        res_rk4 = solvers_pcb.integrate_runge_kutta_4thorder(self.ode_decay, self.t_axis, y_decay)
        
        #Analytical Solution to compare to:
        acc_solution = np.exp(self.t_axis)
        
        
        fig, ax_arr = plt.subplots(2,1, sharex=True)
        fig.suptitle("$dx/dt = x $")
        ax = ax_arr.flatten()

        
        ax[0].plot(self.t_axis, res_rk4, label="Runge-Kutta 4th order", marker = 'x', linestyle='', color='g')
        ax[0].plot(self.t_axis, acc_solution, label="Analytical Solution",color='k')
        ax[0].set_ylabel("$x$")
        
        ax[1].plot(self.t_axis, res_rk4[:,0] - acc_solution, label="Difference", marker = 'x', color='g')
        ax[1].set_xlabel("$t$")
        ax[1].set_ylabel("$\Delta x = x_{rk4}-x(t)$")
        
        

        ax[0].legend()
        plt.tight_layout()
        plt.show()

        plt.close('all')
        fig.clear()
        return res_rk4

    def ode_fall_with_resistance(self, t, y):
        """RHS of the differential equation that describes free fall with Stokes-Resistance:

        y[0] = d/dt v(t) = -gg - (beta / mm) v(t)
        y[1] = d/dt x(t) = v(t)

        Parameter:
        ----------
        gg: gravitational acceleration
        beta: coeffecient of stokes resist
        mm: mass
        """
        dv_dt = - self.gg - (self.beta / self.mm ) * y[0]
        dx_dt = y[0]

        return np.asarray([dv_dt, dx_dt])

    def integrate_ode_fall_ariresistance(self):

        
        y_fall = np.array([0, 0])
        
        v_0 = y_fall[0]
        x_0 = y_fall[1]
        
        res_rk4 = solvers_pcb.integrate_runge_kutta_4thorder(self.ode_fall_with_resistance, self.t_axis, y_fall)        
        res_euler = solvers_pcb.integrate_
        
        acc_solution = np.empty((len(self.t_axis), 2))

        acc_solution[:,0] = ((-self.mm*self.gg)/self.beta)* ( 1- np.exp(-(self.beta/self.mm)*self.t_axis )) + v_0 * np.exp(-(self.beta/self.mm)*self.t_axis )
        acc_solution[:,1] = ( v_0 + (self.mm * self.gg) / self.beta ) * ( self.mm / self.beta) * ( 1- np.exp(-(self.beta/self.mm) * self.t_axis )) - ( (self.mm*self.gg) / self.beta ) * self.t_axis + x_0
        
        
        fig, ax_arr = plt.subplots(2,1)
        ax = ax_arr.flatten()

        ax[0].plot(self.t_axis, acc_solution[:,0], label="$v(t)$ analytical")
        ax[0].plot(self.t_axis, res_rk4[:,0], label="Runge-Kutta 4th order $v(t)$", marker = "x", linestyle="")
        ax[0].set_ylabel("v(t)")

        ax[1].plot(self.t_axis, acc_solution[:,1], label="$x(t)$ analytical")
        ax[1].plot(self.t_axis, res_rk4[:,1], label="Runge-Kutta 4th order $x(t)$", marker = "x", linestyle="")

        ax[1].set_ylabel("$x(t)$")
        ax[1].set_xlabel("$t$") 
        

        ax[0].legend()
        plt.tight_layout()
        plt.show()

        plt.close('all')
        fig.clear()
        return res_rk4

if __name__=="__main__":

    test_cls = test_solver()

    print("Example for the Runge-Kutta 4th for fall with air resistance")
    
    test_cls.integrate_ode_fall_ariresistance()
    
        




