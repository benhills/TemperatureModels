#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 12:21:15 2018

@author: benhills

I want to standardize all of the temperature models.
The models included here, from simplest to most complex are:
    Robin (1955)
    Meyer and Minchew (2018) contact problem
    Perol and Rice (2015)
        Analytic Solution
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import erf

###############################################################################


def Robin_T(Ts,qgeo,H,adot,const,nz=101,melt=True,verbose=True):
    """
    Analytic ice temperature model from Robin (1955)

    Assumptions:
        1) no horizontal advection
        2) vertical advection is linear with depth
        3) firn column is treated as equivalent thickness of ice
        4) If base is warmer than the melting temperature recalculate with new basal gradient
        5) no strain heating

    Parameters
    ----------
    Ts:     float,  Surface Temperature (C)
    qgeo:   float,  Geothermal flux (W/m2)
    H:      float,  Ice thickness (m)
    adot:   float,  Accumulation rate (m/yr)
    const:  class,  Constants
    nz:     int,    Number of layers in the ice column
    melt:   bool,   Choice to allow melting, when true the bed temperature
                    is locked at the pressure melting point and melt rates
                    are calculated

    Output
    ----------
    z:      1-D array,  Discretized height above bed through the ice column
    T:      1-D array,  Analytic solution for ice temperature
    """

    if verbose:
        print('Solving Robin Solution for analytic temperature profile')
        print('Surface Temperature:',Ts)
        print('Geothermal Flux:',qgeo)
        print('Ice Thickness:',H)
        print('Accumulation Rate',adot)

    z = np.linspace(0,H,nz)
    adot/=const.spy
    q2 = adot/(2*(const.k/(const.rho*const.Cp))*H)
    Tb_grad = -qgeo/const.k
    f = lambda z : np.exp(-(z**2.)*q2)
    TTb = Tb_grad*np.array([quad(f,0,zi)[0] for zi in z])
    dTs = Ts - TTb[-1]
    T = TTb + dTs
    # recalculate if basal temperature is above melting (see van der Veen pg 148)
    Tm = const.beta*const.rho*const.g*H
    if melt == True and T[0] > Tm:
        Tb_grad = -2.*np.sqrt(q2)*(Tm-Ts)/np.sqrt(np.pi)*(np.sqrt(erf(adot*H*const.rho*const.Cp/(2.*const.k)))**(-1))
        TTb = Tb_grad*np.array([quad(f,0,zi)[0] for zi in z])
        dTs = Ts - TTb[-1]
        T = TTb + dTs
        M = (Tb_grad + qgeo/const.k)*const.k/const.L
        if verbose:
            print('Melting at the bed: ', round(M*const.spy/const.rho*1000.,2), 'mm/year')
    if verbose:
        print('Finished Robin Solution for analytic temperature profile')
    return z,T

###############################################################################

from scipy.special import lambertw

def Meyer_T(Ts,H,adot,eps_xy,rateFactor,const,nz=101,Tb=0.,lam=0.):
    """
    Meyer and Minchew (2018)
    A 1-D analytical model of temperate ice in shear margins
    Uses the contact problem in applied mathematics

    Assumptions:
        1) horizontal advection is treated as an energy sink
        2) vertical advection is constant in depth (they do some linear analysis in their supplement)
        4) base is at the melting temperature
        5) Melting temperature is 0 through the column

    Parameters
    ----------
    Ts:         float,  Surface Temperature (C)
    H:          float,  Ice thickness (m)
    adot:       float,  Accumulation rate (m/yr)
    eps_xy:     float,  Plane strain rate (m/m)
    rateFactor: func,   function for the rate factor, A in Glen's Law
    const:      class,  Constants
    nz:         int,    Number of layers in the ice column
    Tb:         float,  Basal temperature, at the pressure melting point
    lam:        float,  Paramaterized horizontal advection term
                        Meyer and Minchew (2018) eq. 11

    Output
    ----------
    z:      1-D array,  Discretized height above bed through the ice column
    T:      1-D array,  Analytic solution for ice temperature
    """

    z = np.linspace(0.,H,nz)
    # rate factor
    A = rateFactor(np.array([-10.]))
    # Brinkman Number
    S = 2.*A**(-1./const.n)*(eps_xy)**((const.n+1.)/const.n)
    dT = Tb - Ts
    Br = (S*H**2.)/(const.k*dT)
    # Peclet Number
    Pe = (const.rho*const.Cp*adot*H)/(const.k)
    LAM = lam*H**2./(const.k*dT)
    print('Meyer', A, S)
    # temperature solution is different for diffusion only vs. advection-diffusion
    if abs(Pe) < 1e-3:
        # Critical Shear Strain
        eps_bar = (const.k*dT/(A**(-1/const.n)*H**(2.)))**(const.n/(const.n+1.))
        # Find the temperate thickness
        if eps_xy > eps_bar:
            hbar = 1.-np.sqrt(2./Br)
        else:
            hbar = 0.
        # Solve for the temperature profile
        T = Ts + dT*(Br/2.)*(1.-((z/H)**2.)-2.*hbar*(1.-z/H))
        T[z/H<hbar] = 0.
    else:
        # Critical Shear Strain
        eps_1 = (((0.5*Pe**2.)/(Pe-1.+np.exp(-Pe))+0.5*LAM)**(const.n/(const.n+1.)))
        eps_bar = eps_1 * ((const.k*dT/(A**(-1./const.n)*H**(2.)))**(const.n/(const.n+1.)))
        # Find the temperate thickness
        if eps_xy > eps_bar:
            h_1 = 1.-(Pe/(Br-LAM))
            h_2 = -(1./Pe)*(1.+np.real(lambertw(-np.exp(-(Pe**2./(Br-LAM))-1.))))
            hbar = h_1 + h_2
        else:
            hbar = 0.
        T = Ts + dT*((Br-LAM)/Pe)*(1.-z/H+(1./Pe)*np.exp(Pe*(hbar-1.))-(1./Pe)*np.exp(Pe*((hbar-z/H))))
        T[z/H<hbar] = 0.
    return z,T

###############################################################################

def PerolRiceAnalytic(Ts,adot,H,eps_xy,rateFactor,const,nz=101,T_ratefactor=-10.):
    """
    Perol and Rice (2015)
    Analytic Solution for temperate ice in shear margins (equation #5)

    Assumptions:
        1) Bed is at the melting point
        2) All constants are temperature independent (rate factor uses T=-10)

    Parameters
    ----------
    Ts:         float,  Surface Temperature (C)
    adot:       float,  Accumulation rate (m/yr)
    H:          float,  Ice thickness (m)
    eps_xy:     float,  Plane strain rate (m/m)
    rateFactor: func,   function for the rate factor, A in Glen's Law
    const:      class,  Constants
    nz:         int,    Number of layers in the ice column
    T_ratefactor float, Temperature input to the rate factor function, A(T)

    Output
    ----------
    z:          1-D array,  Discretized height above bed through the ice column
    T:          1-D array,  Analytic solution for ice temperature
    """

    # Height
    z = np.linspace(0,H,nz)
    # Peclet Number
    Pe = adot*H/(const.k/(const.rho*const.Cp))
    # Rate Factor
    A = rateFactor(np.array([T_ratefactor]))
    # Strain Heating
    S = 2.*A**(-1./const.n)*(eps_xy/2.)**((const.n+1.)/const.n)
    print('Perol', A, S)
    # Pressure Melting Point at Bed
    Tm = const.beta*const.rho*9.81*H
    # Empty Array for Temperatures, then loop through all z's
    T = np.empty_like(z)
    for i in range(len(z)):
        # Two functions to be integrated
        def f1(lam):
            return (1.-np.exp(-lam*Pe*z[i]**2./(2.*H**2.)))/(2.*lam*np.sqrt(1.-lam))
        def f2(lam):
            return (1.-np.exp(-lam*Pe/2.))/(2.*lam*np.sqrt(1.-lam))
        # Calculate temperature profile
        T[i] = Tm + (Ts-Tm)*erf(np.sqrt(Pe/2.)*(z[i]/H))/erf(np.sqrt(Pe/2.)) - \
            S*H**2./(const.k*Pe) * \
            (quad(f1,0.,1.)[0] - (erf(np.sqrt(Pe/2.)*(z[i]/H))/erf(np.sqrt(Pe/2.)))*quad(f2,0.,1.)[0])
    return z,T
