#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 16:22:31 2018

@author: benhills
"""

import numpy as np

###############################################################################

def rateFactorCuffPat(temp,const,P=0.):
    """

    Rate Facor function for ice viscosity, A(T)
    Cuffey and Paterson (2010), equation 3.35

    Parameters
    --------
    temp:   float,  Temperature
    const:  class,  Constants
    P:      float,  Pressure

    Output
    --------
    A:      float,  Rate Factor, viscosity = A^(-1/n)/2

    """
    # create an array for activation energies
    Q = const.Qminus*np.ones_like(temp)
    Q[temp>-10.] = const.Qplus
    # Convert to K
    T = temp + const.T0
    # equation 3.35
    A = const.Astar*np.exp(-(Q/const.R)*((1./(T+const.beta*P))-(1/const.Tstar)))
    return A

###############################################################################

def rateFactorVanDerVeen(temp,const):
    """

    Rate Facor function for ice viscosity, A(T), B(T)
    van der Veen (2013), eq 2.14, 2.15

    Parameters
    --------
    temp:   float,  Temperature
    const:  class,  Constants

    Output
    --------
    A:      float,  Rate Factor, eta = A^(-1/n)/2
    B:      float,  Rate Factor, eta = B^n

    """
    # Convert to K
    T = temp + const.Tr
    # Equation 2.14
    A = const.A0*np.exp(-const.Q/(const.R*T)+3*const.C/((const.Tr-T)**const.Kk))
    # Equation 2.15
    B = const.B0*np.exp(const.T0/T-const.C/((const.Tr-T)**const.Kk))
    # Returns A in kPa-3 yr-1, B in kPa yr1/3
    return A,B
