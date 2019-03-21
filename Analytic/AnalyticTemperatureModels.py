#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 10:49:17 2018

@author: benhills
"""

import numpy as np
from scipy.special import erfc

# Analytic solutions to conduction problems
# All are from Carslaw and Jaeger (1959)

###############################################################################

### Temperature near a constant boundary ###
# The infinite and semi-infinite solid
# Carslaw and Jaeger Ch. 2

def erfSolution(dT,rho,C,k,t,x):
    """
    Carslaw and Jaeger Section 2.4
    Boundary step change in temperature, held constant at boundary

    Parameters
    --------
    dT
    rho
    C
    k
    t
    x

    Output
    --------
    T

    """
    # diffusivity
    alpha = k/(rho*C)
    # equation 2.4-7
    T = dT*erfc(abs(x)/(2.*np.sqrt(alpha*t)))
    return T

def harmonicSurface(Tmaat,Tamp,rho,C,k,t,x,w):
    """
    Carslaw and Jaeger Section 2.6
    Surface Boundary Temperature Harmonic Function

    Parameters
    --------
    Tmaat
    Tamp
    rho
    C
    k
    t
    x
    w

    Output
    --------
    T

    """
    # diffusivity
    alpha = k/(rho*C)
    # equation 2.6-8
    T = Tmaat + Tamp * np.exp(-x*np.sqrt(w/(2.*alpha))) * np.cos((w*t)-x*np.sqrt(w/(2*alpha)))
    return T

###############################################################################

def harmonicAdvection(Tmaat,Tamp,rho,C,k,t,x,w,vel):
    """
    Survace boundary temperature, harmonic function AND advection ###
    Logan and Zlotnic (1995)

    Parameters
    --------
    Tmaat
    Tamp
    rho
    C
    k
    t
    x
    w
    vel

    Output
    --------
    T

    """

    # diffusivity
    alpha = k/(rho*C)
    # set up with variables from LZ (1995) eq. 4.4
    phi = w/alpha
    psi = vel**2/(4*alpha**2)
    mu = np.sqrt((np.sqrt(phi**2+psi**2)+psi)/2.)
    rho = np.sqrt((np.sqrt(phi**2+psi**2)-psi)/2.)
    # LZ (1995) eq. 4.6
    T = Tmaat + Tamp * np.exp((vel/(2*alpha)-mu)*x) * np.cos(w*t-rho*x)
    return T

###############################################################################

def parallelPlates(dT,rho,C,k,t,x,l,N=1000):
    """
    Temperature between two plates

    Carslaw and Jaeger Ch. 3
    Linear flow of heat in the solid bounded by two parallel planes

    Parameters
    --------
    dT:     float,  Temperature difference between boundaries and internal material
    rho:    float,  Material density
    C:      float,  Material heat capacity
    k:      float,  Material thermal conductivity
    t:      float,  Time after initiation
    x:      float,  Distance from left hand boundary
    l:      float,  Distance between plates
    N:      int,    Number of iterations to sum over

    Output
    --------
    T:      float,  Temperature at output time and location


    """

    # diffusivity
    alpha = k/(rho*C)
    # infinite sum for equation 3.4-2
    infsum = 0.
    for n in range(N):
        infsum += (((-1.)**n)/(2*n+1))*np.exp(-alpha*(2.*n+1)**2.*np.pi**2.*t/(4.*l**2.))*np.cos((2*n+1)*np.pi*x/(2*l))
    # equation 3.4-2
    T = dT - (4.*dT/np.pi)*infsum
    return T

###############################################################################

def InstSource(Q,rho,C,k,t,x,y,z,x_=0.,y_=0.,z_=0.,dim=3):
    """
    Instantaneous Source

    Carslaw and Jaeger Ch. 10, pg 256-259
    The use of sources and sinks in cases of variable temperature

    Parameters
    ----------
    Q:          float,  Source magnitude
    rho:        float,  Material density
    C:          float,  Material heat capacity
    k:          float,  Material thermal conductivity
    t:          float,  Time after source input or 1-d array for times
    x,y,z:      float,  3-d space
    x_,y_,z_:   float,  Centerpoint for source
    dim:        int,    # of dimensions

    Output
    -------
    T:          float,  Resulting temperature

    """
    # Define the diffusivity
    alpha = k/(rho*C)
    # Point source, CJ (1959) pg. 256
    if dim==3:
        return Q/(8.*(np.pi*alpha*t)**(3/2.))*np.exp(-((x-x_)**2.+(y-y_)**2.+(z-z_)**2.)/(4.*alpha*t))
    # Linear source, CJ (1959) pg. 258
    elif dim==2:
        return Q/(4.*np.pi*alpha*t)*np.exp(-((x-x_)**2.+(y-y_)**2.)/(4*alpha*t))
    # Planar source, CJ (1959) pg. 259
    elif dim==1:
        return Q/(2.*(np.pi*alpha*t)**(1/2.))*np.exp(-((x-x_)**2.)/(4*alpha*t))

