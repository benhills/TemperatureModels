#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created Mar 2016

@author: benhills
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def tempFiniteDiff_2D(dx,dz,dt,T_init,const):
    """

    TODO: !!! This will need major changes before use !!!

    This is a thermal diffusion model with a heat source that can move in the
    x and z directions through time. The heat source can also change in magnitude
    over time. A Crank-Nicolson Finite Difference scheme is used with Dirichlet
    boundary conditions

    The original point was to use this alongside PEST to invert for the heat
    source observed in Patrick's thermistor string.

    Parameters
    ---------
    dx = .25
    dz = .25
    dt = 0.1
    T_init:     float,  Initial temperature
    const


    Output
    ---------
    """

    X,Z,N,nx,nz = Domain(zi=0,zf=25,dz=dz,xi=-10,xf=10,dx=dx)
    T = T_init*np.ones_like(X)

    LMat, RMat, Sz = Operator(N,nx,nz,dt,dx,dz,const.k)
    S = Source(Mf,dMf,dMf2,t,t0,dt,x0,dx0,z0,dz0,T,X,Z,const)

    print('\nBegin Thermal Diffusion')
    T = np.reshape(T,N)             # reshape to a 1-D array to solve the system of equations
    print('Max Temperature = ', max(T))
    un = T.copy()

    # If the Robin BC is not 0.0 (check before using)
    b = 0.0*T
    b[0:nx+1] = 4*Sz*dx*c

    rhs = RMat*un+b                 # right hand side for solve
    T = spsolve(LMat,rhs)           # solve the system of equations
    T = T.reshape(nz,nx)            # back to original shape
    return T
    return T

##################### Domain Definition #####################################3

# Define arrays (X and Z) for x and z spatial values, the size of each array
# will be equivalent to the size of the array of temperature values
def Domain(zi,zf,dz,xi,xf,dx):
    print('Creating the Domain')
    z = np.arange(zi,zf+dz,dz)                      #Ice Depth (m)
    x = np.arange(xi,xf+dx,dx)                      #Width of the grid (m)
    nx = len(x)                                     #Number of nodes in x-direction
    nz = len(z)                                     #Number of nodes in z-direction
    N = nx*nz                                       #Size for operator matrix
    X = np.tile(x,(nz,1))
    Z = np.transpose(np.tile(z,(nx,1)))             #An N-sized grid for both x and z
    return  X,Z,N,nx,nz

########################## Setting Up the Operator Matrix ##############################

# Define a function to write a sparse matrix
def Sp_Matr(N,nx,diag,k1,k2,k3,k4):
    A = sparse.lil_matrix((N, N))           #Function to create a sparse Matrix
    A.setdiag((diag)*np.ones(N))            #Set the diagonal
    A.setdiag((k1)*np.ones(N-1),k=1)        #Set the first upward off-diagonal.
    A.setdiag((k2)*np.ones(N-1),k=-1)       #Set the first downward off-diagonal
    A.setdiag((k3)*np.ones(N-nx),k=nx)      #Set for diffusion from above node
    A.setdiag((k4)*np.ones(N-nx),k=-nx)     #Set for diffusion from below node
    return A

# Write the operator using Sp_Matr from above
def Operator(N,nx,nz,dt,dx,dz,k):
    print('Writing the Operator Matrix')

    #Diffusion constants
    Sx = k*dt/(dx**2.)
    Sz = k*dt/(dz**2.)
    # In order to use the Crank-Nicolson formulation we need both right and left
    # matrices for the previous and current time steps
    # First define these matrices based on the heat equation, then apply Boundary Conditions
    Ldiag = 2. + 2.*Sx + 2.*Sz
    Rdiag = 2. - 2.*Sx - 2.*Sz
    LMat = Sp_Matr(N,nx,Ldiag,-Sx,-Sx,-Sz,-Sz)
    RMat = Sp_Matr(N,nx,Rdiag,Sx,Sx,Sz,Sz)

    ### Enforce Diriclet BC at the S,W,E boundaries ###
    #South
    for i in range(-nx,0):
        LMat[i,i],LMat[i,:i],LMat[i,i+1:] = 1.,0.,0.
        RMat[i,i],RMat[i,:i],RMat[i,i+1:] = 1.,0.,0.
    #East and West
    for i in range(1,nz):
        LMat[i*nx,i*nx],LMat[i*nx,:i*nx],LMat[i*nx,i*nx+1:] = 1.,0.,0.
        LMat[i*nx-1,i*nx-1],LMat[i*nx-1,:i*nx-1],LMat[i*nx-1,i*nx:] = 1.,0.,0.
        RMat[i*nx,i*nx],RMat[i*nx,:i*nx],RMat[i*nx,i*nx+1:] = 1.,0.,0.
        RMat[i*nx-1,i*nx-1],RMat[i*nx-1,:i*nx-1],RMat[i*nx-1,i*nx:] = 1.,0.,0.
    # Very last node
    LMat[-1,-1],LMat[-1,:-1] = 1.,0.
    RMat[-1,-1],RMat[-1,:-1] = 1.,0.

    ### Enforce Neumann BC at the North boundary so that flux is always 0.0 ###
    for i in range(0,nx+1):
        LMat[i,i] = (2+2*Sx+2*Sz)
        RMat[i,i] = (2-2*Sx-2*Sz)
        LMat[i,i-1],LMat[i,i+1],LMat[i,i-nx] = -Sx,-Sx,-2*Sz
        RMat[i,i-1],RMat[i,i+1],RMat[i,i-nx] = Sx,Sx,2*Sz

    print('Written, Shape = ', np.shape(LMat))
    return LMat, RMat, Sz

####################### Heat Source and Diffusion ###############################

def Source(Mf,dMf,dMf2,t,t0,dt,x0,dx0,z0,dz0,T,X,Z,const):
    print('\nApply Heat Source')

    # find the current location and magnitude of the heat source
    x1 = x0 + dx0*(t-t0)
    z1 = z0 + dz0*(t-t0)
    # Magnitude is based on the formulation in Van der Veen (2013)
    Src = dt*(Mf*const.L/(const.rho*const.C) + (dMf*const.L/(const.rho*const.C))*(t-t0) + (dMf2*const.L/(const.rho*const.C))*(t-t0)**2)

    # If the event has not started yet, becomes a heat sink, or moves out of the domain
    # set it equal to zero.
    if t < t0:
        Src = 0.0
    if Src < 0.0:
        Src = 0.0
    if z0 < 0.0:
        Src = 0.0

    # Set the source array "S" equal to T so that it is the same shape.
    S = T*0.0
    # Add the heat source Src at the x and z locations x1 and z1 (calculated above)
    S[np.where(abs(Z[:,0]-z1)==min(abs(Z[:,0]-z1)))[0][0],
      np.where(abs(X[0]-x1)==min(abs(X[0]-x1)))[0][0]] = Src

    print('Heat Source Magnitude = ', Src)
    print('Heat Source Location, x =', x1, ' z =', z1)
    return S
