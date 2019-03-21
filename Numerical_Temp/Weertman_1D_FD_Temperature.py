#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 08:55:55 2018

@author: benhills
"""

# Import necessary libraries
import numpy as np
from scipy.special import erf
from scipy.optimize import least_squares
from scipy import sparse

def Weertman_T(Ts,qgeo,H,adot,const,dTs=[0.,0.],dH=[0.,0.],da=[0.,0.],v_surf=[0.,0.],
               eps_xy=0.,nz=101,conv_crit=1e-4):
    """
    1-D finite difference model for ice temperature based on
    Weertman (1968)

    Assumptions:
        1) Initialize from Weertman's version of the Robin (1955) analytic solution.
            This has been checked against the Robin_T function and gives consistent
            results to 1e-14.
        2) Finite difference solution
        3) Horizontal velocity...
        4) Vertical velocity...
        5) Strain rates...

    Parameters
    ----------
    Ts:         float,      Surface Temperature                         [C]
    qgeo:       float,      Geothermal flux                             [W/m2]
    H:          float,      Ice thickness                               [m]
    adot:       float,      Accumulation rate                           [m/yr]
    const:      class,      Constants
    dTs:        2x1 array,  Change in air temperature over distance x/y [C/km]
    dH:         2x1 array,  Thickness gradient in x/y directions        [m/km]
    da:         2x1 array,  Accumulation gradient in x/y directions     [m/yr/km]
    v_surf:     2x1 array,  Surface velocity in x/y directions          [m/yr]
    eps_xy:     float,      Plane strain rate                           [m/m]
    nz:         int,        Number of layers in the ice column
    conv_crit:  float,      Convergence criteria, maximum difference between
                            temperatures at final time step

    Output
    ----------
    z:          1-D array,  Discretized height above bed through the ice column
    T_weertman: 1-D array,  Numerical solution for ice temperature
    T_robin:    1-D array,  Analytic solution for ice temperature
    T_diff:     1-D array,  Convergence profile (i.e. difference between final and t-1 temperatures)

    """

    # Height above bed array
    z = np.linspace(0,H,nz)
    # accumulation rate to m/sec
    adot/=const.spy

    ###########################################################################

    ### Start with the analytic 'Robin Solution' as an initial condition ###

    # Vertical Velocity
    v_z_surf = adot+v_surf[0]*dH[0]+v_surf[1]*dH[1]
    # Characteristic Diffusion Length, Weertman (1968) eq. 2
    L = (2.*(const.k/(const.rho*const.Cp))*H/v_z_surf)**.5
    # Weertman (1968) eq. 1
    T_robin = Ts + (np.pi/4.)**.5*L*(-qgeo/const.k)*(erf(z/L)-erf(H/L))

    ###########################################################################

    ### Optimize the rate factor to fit the surface velocity ###

    # Shear Stress by Lamellar Flow (van der Veen section 4.2)
    tau_xz = const.rho*const.g*(H-z)*abs(dH[0])
    tau_yz = const.rho*const.g*(H-z)*abs(dH[1])

    # Function to Optimize
    def surfVelOpt(C):
        # Change the coefficient so that the least_squares function takes appropriate steps
        # (there is likely a better way to do this)
        C_opt = C*1e-13
        # Shear Strain Rate, Weertman (1968) eq. 7
        eps_xz = C_opt*np.exp(-const.Qminus/(const.R*(T_robin+const.T0)))*tau_xz**const.n
        vx_opt = np.trapz(eps_xz,z)
        return abs(vx_opt-v_surf[0])*const.spy
    # Get the final coefficient value
    res = least_squares(surfVelOpt, 1)
    C_fin = res['x']*1e-13

    # Final Strain Rates, Weertman (1968) eq. 7
    eps_xz = C_fin*np.exp(-const.Qminus/(const.R*(T_robin+const.T0)))*tau_xz**const.n
    eps_yz = C_fin*np.exp(-const.Qminus/(const.R*(T_robin+const.T0)))*tau_yz**const.n
    # Horizontal Velocity (integrate the strain rate through the column)
    v_x = np.empty_like(z)
    v_y = np.empty_like(z)
    for i in range(len(z)):
        v_x[i] = np.trapz(eps_xz[:i+1],z[:i+1])
        v_y[i] = np.trapz(eps_yz[:i+1],z[:i+1])

    ###########################################################################

    ### Calculate Strain Heat Production and Advective Sources ###

    # effective stress and strain rate (van der Veen eq. 2.6/2.7)
    tau_e = np.sqrt((2.*tau_xz**2. + 2.*tau_yz**2.)/2.)
    eps_e = np.sqrt((2.*eps_xz**2. + 2.*eps_yz**2.)/2.)
    # strain heat term (K s-1)
    Q = (eps_e*tau_e)/(const.rho*const.Cp)

    # Horizontal Temperature Gradients, Weertman (1968) eq. 6b
    dTdx = dTs[0] + (T_robin-Ts)/2.*(1./H*dH[0]-(1/adot)*da[0])
    dTdy = dTs[1] + (T_robin-Ts)/2.*(1./H*dH[1]-(1/adot)*da[1])
    # Advection Rates (K s-1)
    Adv_x = -v_x*dTdx
    Adv_y = -v_y*dTdy

    # Final source term
    Sdot = Q + Adv_x + Adv_y

    ###########################################################################

    ### Finite Difference Scheme ###

    # Initial Condition from Robin Solution
    T = T_robin.copy()
    dz = np.mean(np.gradient(z))
    Tgrad = -qgeo/const.k
    v_z = v_z_surf*z/H

    # Stability
    dt = 0.5*dz**2./(const.k/(const.rho*const.Cp))
    if max(v_z)*dt/dz > 1.:
        raise ValueError("Numerically unstable, choose a smaller time step")

    # Stencils
    diff = (const.k/(const.rho*const.Cp))*(dt/(dz**2.))
    A = sparse.lil_matrix((nz, nz))           # create a sparse Matrix
    A.setdiag((1.-2.*diff)*np.ones(nz))            #Set the diagonal
    A.setdiag((1.*diff)*np.ones(nz),k=-1)            #Set the diagonal
    A.setdiag((1.*diff)*np.ones(nz),k=1)            #Set the diagonal
    B = sparse.lil_matrix((nz, nz))           # create a sparse Matrix
    for i in range(len(z)):
        adv = (-v_z[i]*dt/dz)
        B[i,i] = adv
        B[i,i-1] = -adv

    # Boundary Conditions
    # Neumann at bed
    A[0,1] = 2.*diff
    B[0,:] = 0.
    # Dirichlet at surface
    A[-1,:] = 0.
    A[-1,-1] = 1.
    B[-1,:] = 0.

    # Source Term
    Sdot[0] = -2*dz*Tgrad*diff/dt
    Sdot[-1] = 0.

    ###########################################################################

    ### Iterations and Output ###

    # Iterate until convergence
    i = 0
    T_diff = 0.
    while i < 1 or sum(abs(T_diff)) > conv_crit:
        T_new = A*T - B*T + dt*Sdot
        T_diff = T_new-T
        print(sum(abs(T_diff)))
        T = T_new
        i += 1

    T_weertman = T
#
    return z,T_weertman,T_robin,T_diff
