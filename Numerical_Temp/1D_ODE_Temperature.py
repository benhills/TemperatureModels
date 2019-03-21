#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 08:55:55 2018

@author: benhills

"""

import numpy as np
from IceViscosity import rateFactorCuffPat
from scipy.special import erf
from scipy.optimize import least_squares


def tempPerolRiceShooting(Ts,qgeo,H,adot,eps_xy,surf_vel,nz=101)
    """
    1-D ODE model for ice temperature based on
    Perol and Rice (2015)

    Uses Runge Kutta and Shooting Methods

    Assumptions:
    """

    z = np.linspace(0,H,nz)

    ### ODE Solver ###

    return z,T




"""

### "Robin Solution" ###
# Vertical Velocity
v_z = adot+v_surf[0]*dH[0]+v_surf[1]*dH[1]
# Characteristic Diffusion Length, Weertman (1968) eq. 2
L = (2.*(const.k/(const.rho*const.Cp))*H/v_z)**.5
# Weertman (1968) eq. 1
T_robin = Ts + (np.pi/4.)**.5*L*(qgeo/const.k)*(erf(z/L)-erf(H/L))
T = T_robin




### Weertman Solution ###

### Optimize the rate factor to fit the surface velocity ###

# Shear Stress by Lamellar Flow (van der Veen section 4.2)
tau_xz = const.rho*constU.g*(H-z)*abs(dH[0])
tau_yz = const.rho*constU.g*(H-z)*abs(dH[1])

# Activation Energy
Qstar = 0.6*1.60218e-19*6.02e23
# Function to Optimize
def surfVelOpt(C):
    # Change the coefficient so that the least_squares function takes appropriate steps
    # (there is likely a better way to do this)
    C_opt = C*1e-13
    # Shear Strain Rate, Weertman (1968) eq. 7
    eps_xz = C_opt*np.exp(-Qstar/(constU.R*(T+constU.T0)))*tau_xz**const.n
    vx_opt = np.trapz(eps_xz,z)
    return abs(vx_opt-v_surf[0])*const.spy
# Get the final coefficient value
res = least_squares(surfVelOpt, 1)
C_fin = res['x']*1e-13

# Final Strain Rates, Weertman (1968) eq. 7
eps_xz = C_fin*np.exp(-Qstar/(constU.R*(T+constU.T0)))*tau_xz**const.n
eps_yz = C_fin*np.exp(-Qstar/(constU.R*(T+constU.T0)))*tau_yz**const.n
# Horizontal Velocity (integrate the strain rate through the column)
v_x = np.empty_like(z)
v_y = np.empty_like(z)
for i in range(len(z)):
    v_x[i] = np.trapz(eps_xz[:i+1],z[:i+1])
    v_y[i] = np.trapz(eps_yz[:i+1],z[:i+1])

### Calculate Strain Heat Production and Advective Sources ###

# effective stress and strain rate (van der Veen eq. 2.6/2.7)
tau_e = np.sqrt((2.*tau_xz**2. + 2.*tau_yz**2.)/2.)
eps_e = np.sqrt((2.*eps_xz**2. + 2.*eps_yz**2.)/2.)
# strain heat term (K s-1)
Q = (eps_e*tau_e)/(const.rho*const.Cp)

# Horizontal Temperature Gradients, Weertman (1968) eq. 6b
dTdx = dTs[0] + (T-Ts)/2.*(1./H*dH[0]-(1/adot)*da[0])
dTdy = dTs[1] + (T-Ts)/2.*(1./H*dH[1]-(1/adot)*da[1])
# Advection Rates (K s-1)
Adv_x = -v_x*dTdx
Adv_y = -v_y*dTdy

# Final source term
Sdot = Q# + Adv_x + Adv_y

v_z = adot*z/H

#"""
