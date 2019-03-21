# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:52:13 2016

@author: ben

# This is a 1D Finite Element Solver for ice temperature profiles.

"""

import numpy as np
import matplotlib.pyplot as plt
from dolfin import *

import sys
sys.path.append('/Users/benhills/Documents/PythonScripts/')
from Constants import *

################################################################################################################################

### Constants ###

const = constantsTempCuffPat()

# General
N = 1000                # Size of mesh
dt = 100.*const.spy    # Time step in years


"""
# Constants from Perol
Margins = ['A','WB1','WB2','W Narrows','W Plain','TWB1','TWB2','C','TC1','TC2','D','TD1','TD2','TD3','E','TE']
eps_xy_arr = np.array([4.2,7,9.5,13.5,5.1,3.8,4.0,1.,1.4,0.9,5.8,2.5,5.4,2.2,8.1,5.5])*(1e-2/const.spy)
H_arr = np.array([1242.,1205.,985.,846.,735.,2188.,1538.,1802.,1802.,2196.,888.,1952.,1412.,1126.,916.,1177.])

#for Mi in [1]:#range(len(Margins)):
eps_xy = 13.5*1e-2/const.spy#eps_xy_arr[Mi]
H = 846.# H_arr[Mi]
bed, surf = 0, H
adv = -.1/const.spy
q_geo = 0.042#.0513       # Geothermal heat flux (W m-2) (Dahl-Jensen 1998)
alpha = 0.01
v_x = 100./const.spy
"""

# NEGIS values
eps_xy = 0.006/const.spy
H = 2600.
bed, surf = 0,H
adv = -.1/const.spy
v_x = 40./const.spy
# calculate the surface slope using the surface velocity
alpha = ((4.*v_x/(const.Astar*H**4.))**(1./const.n))/(const.rho*const.g)
Ts = -32.

################################################################################################################################

# Mesh
Mesh = IntervalMesh(N,bed,surf)
V = FunctionSpace(Mesh,'CG',1)

# Pressure-melting point
P = Expression('rho*g*(H-x[0])',rho=const.rho,g=const.g,H=H)
Tm = project(P*const.beta,V)

# Temperatures
Tbed, Tsurf = P(0)*const.beta+const.T0, Ts+const.T0
Tr = -10.+const.T0
T_1 = project(Expression('x[0]*(Tsurf-Tbed)/H + Tbed',H=H,Tsurf=Tsurf,Tbed=Tbed),V)

################################################################################################################################

### Define the Boundaries ###

class Surf(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 0.5*(surf-bed)and on_boundary
bc_surf = DirichletBC(V, Tsurf, Surf())
class Bed(SubDomain):
    def inside(self,x,on_boundary):
        return x[0] < 0.5*(surf-bed) and on_boundary
bc_bed = DirichletBC(V, Tbed, Bed())
bcs = [bc_surf,bc_bed]

# Instantiate sub-domain instances
bottom = Bed()
boundaries = FacetFunction("size_t", Mesh, 1)
boundaries.set_all(0)
# mark ds(1) as bed to be used in the variational problem with q_geo
bottom.mark(boundaries, 1)
ds = Measure("ds")[boundaries]

################################################################################################################################

# Define test and trial function
u = Function(V)
v = TestFunction(V)

# Temperature-dependent functions from Cuffey and Paterson 9.1 and 9.2
k = 9.828*exp(-0.0057*u)
C = 152.5+7.122*u
kappa = k/(C*const.rho)
# Rate factor from Cuffey and Paterson 3.35
A = exp((-const.Qminus/const.R)*((1./(T_1+Tm))-(1./(Tr+Tm))))

# Vertical velocity
vel = Expression('adv*x[0]/H',adv=adv,H=H)
# xz stress by Lamellar Flow (van der Veen section 4.2)
#tau_xz = Expression('rho*g*(H-x[0])*alpha',rho=const.rho,g=const.g,H=H,alpha=alpha)
eps_hold = const.Astar*(const.rho*const.g*(H-np.linspace(0,H,N+1))*alpha)**const.n
eps_xz = Function(V)
eps_xz.vector()[:] = eps_hold[:]
#eps_xz = v_x/H
# xz and xy strain heating term from Perol and Rice (2015) eq. 3
s_xz = 2.*const.Astar**(-1./const.n)*A**(-1./const.n)*(eps_xz/2.)**((1.+const.n)/const.n)
s_xy = 2.*const.Astar**(-1./const.n)*A**(-1./const.n)*(eps_xy/2.)**((1.+const.n)/const.n)
# source is the total
source = s_xz+s_xy
# cap at melting temp from Perol and Rice (2015) eq. 6
heaviside = .5+1./np.pi*atan(-100.*(u-Tm-const.T0))
source*=heaviside

F1 = ((u-T_1)/dt + inner(vel,u.dx(0)))*v*dx + inner(kappa*grad(u),grad(v))*dx - source/(C*const.rho)*v*dx

plot(T_1-const.T0-Tm,vmin=min(u.vector())-const.T0,vmax=max(u.vector())-const.T0)

for i in np.arange(1000):
    solve(F1==0,u,bcs)#,solver_parameters={"newton_solver":{"relative_tolerance": 1e-6}})
    T_1.assign(u)

    if i > 990:
        # Plot
        plot(u-const.T0-Tm,vmin=min(u.vector())-const.T0,vmax=max(u.vector())-const.T0)

zs_out = np.linspace(0,H,N+1)
np.save('NEGIS_Stream_Temps',np.array([zs_out,np.array([u(z)-const.T0 for z in zs_out])]))
