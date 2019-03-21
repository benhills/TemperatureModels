#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 11:31:28 2016

@author: ben
"""

"""
This is a sample model for advection/diffusion of enthalpy.
"""

import numpy as np
from time import sleep
from dolfin import *

### Mesh ###

# For now structured rectangle
xmin,xmax = -5,5
ymin,ymax = -10,0
mesh = RectangleMesh(Point(xmin,ymin),Point(xmax,ymax),25,25)
V = FunctionSpace(mesh,'CG',1) 

################################################################################################################################

### Variables ###

# Define the lithostatic pressure
P = Expression('rho*g*max(0.,x[1])',rho=const.rho, g=const.g)
# Define temperature at the melting point
Tm = Expression('T0-gamma*P',T0=const.T0, gamma=const.gamma, P=P)  

# Define the enthalpy variable
H = Function(V)
H0 = Expression('(temp-Tm)*C',temp=const.T0+Tinit,Tm=Tm,C=const.C)
H_1 = interpolate(H0,V)

# Temperature
T = Expression('min((H)/C + Tm,Tm)',H=H,C=const.C,Tm=Tm)
# Water Content
wc = Expression('max(0.,(H)/L)',H=H,L=const.L)

################################################################################################################################

### Parameterizations ###

# Define a function for the enthalpy diffusivity
# This uses an arctan as a type of conditional to select between k/(C*rho), the cold ice diffusivity, and nu/rho, the temperate diffusivity
kappa_ice = 0.5*(const.k/(const.C*const.rho)-const.nu/const.rho)*(1.-((2./np.pi)*atan(.01*(H))))+const.nu/const.rho

# create a velocity expression that will be set equal to the ablation rate
vel = Expression('melt',melt=-0.05/const.sph)   # melt rate is in meters per hour (the time step may need to change if this is too fast)

# Source term
source = Expression('x[0] > -.5 && x[0] < .5 && x[1] < -4.5 && x[1] > -5.5 ? H_source : 0.', H_source=H_source)

################################################################################################################################

### Boundary Conditions ###
# TODO: change the boundary conditions for a more realistic scenario

# Surface BC
class Surface(SubDomain):
    def inside(self,x,on_boundary):
        return x[1] > ymax-0.1 and x[0] < xmax and x[0] > xmin and on_boundary
bc_surf = DirichletBC(V, -10.*const.C, Surface())                

# Basal BC
class Bed(SubDomain):
    def inside(self,x,on_boundary):
        return x[1] < ymin+0.1 and x[0] < xmax and x[0] > xmin and on_boundary
bc_bed = DirichletBC(V, -10.*const.C, Bed())

################################################################################################################################

### Solver Setup ###

# Define test and trial function
u = TrialFunction(V)
v = TestFunction(V)   
 
# Set up the variational form, (see Brinkerhoff 2013)
a = u*v*dx + dt*kappa_ice*inner(nabla_grad(u), nabla_grad(v))*dx - dt*dot(vel,u.dx(1))*v*dx
L = (H_1 + dt*source/const.rho)*v*dx
A = assemble(a)

################################################################################################################################

### Time Stepping ###

ts = np.arange(0,100)
for t in ts:
    # Reset the advection rate based on melting at the probe
    # TODO: allow this term to evolve with the melt rate at the probe
    #vel.melt = (IceSurf[ind+1] - IceSurf[ind])/const.spd

    # Solve problem
    b = assemble(L)
    bc_surf.apply(A,b)
    bc_bed.apply(A,b)
    solve(A,H.vector(),b)
    
    # Update RHS
    H_1.assign(H)

T.H = H
wc.H = H

Temps = interpolate(T,V)
Water = interpolate(wc,V)

################################################################################################################################

### Plot Figures ###

import pylab as plt

# Mesh and Source

plt.figure()

ax1 = plt.subplot(121)
pm = plot(mesh)
plt.title("Mesh")

ax2 = plt.subplot(122)
ps = plot(interpolate(source,V))
plt.title("Source")

# Temperature and Water Content

plt.figure()

ax1 = plt.subplot(121)
p1 = plot(Temps-const.T0)
p1.set_cmap("RdYlBu_r")
p1.set_clim(-15.,0.)
plt.title("Temperature")
plt.colorbar(p1)

ax2 = plt.subplot(122)
p2 = plot(Water)
p2.set_cmap("Blues")
p2.set_clim(0.,.1)
plt.title("Water Content")
plt.colorbar(p2)