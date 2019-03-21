#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:52:13 2016

@author: ben

This is a heat transfer model which was written to help us understand
warming mechanisms in the ablation zone. The temperature is initiated to
measured data, then an advection-diffusion equation is solved at each time step.
I use a Lagrangian reference frame, so the profile is tracking down-glacier
through time. The mesh shrinks to account for ablation at the surface, and the
model is run until the ice thickness is 0.0
"""

import numpy as np
from dolfin import *
from scipy.optimize import leastsq

### Constants ###
### (Van der Veen 2013) ###
class Constants(object):
    def __init__(self):
        self.g = 9.81                            #Gravity (m/s2)
        self.spy  = 60.*60.*24.*365.24           #Seconds per year
        self.gamma = 7.4e-8                        #Clausius-Clapeyron (K Pa-1) From van der Veen p. 209
        # pg 144 Van der Veen - from Yen CRREL 1981
        self.rho = 917.                         #Bulk density of ice (kg m-3)
        self.rhow = 1000.                       # density of water
        self.k = 2.1                          #Conductivity of ice (J m-1 K-1 s-1)
        self.C = 2097.                        #Heat capacity of ice (J kg-1 K-1) - ** Van der Veen uses 2097 but see Tr and Aschwanden 2012)
        self.L = 3.335e5                        #Latent heat of fusion (J kg-1)
        # From other sources
        self.nu = 1.045e-4                  # moisture diffusivity for numerical stability, Brinkerhoff (2013)
        self.T0 = 273.15                         #Reference Tempearature, triple point for water
        self.tol = DOLFIN_EPS              #Tolerance for numerical issues
        self.R = 8.321           # the gas constant
        self.q_geo = .0513       # Geothermal heat flux (W m-2) (Dahl-Jensen 1998)

# initialize constants class
const = Constants()

### Problem-specific parameters ###
Uvel = 100          # horizontal velocity (m/yr)
# Lapse rate constants
Abl_lapse = .0038   # Change in the ablation rate with elevation, van de Wal (2012)
Abl_start = -4.22   # Ablation rate at 0 ice thickness, van de Wal (2012)
T_lapse = -.0068    # Change in surface temperature with elevation, Fausto (2009)
T_start = -4.5           # Surface temperature at 0 ice thickness, -5 at Kanger van As (2012) -4.5 from Promice data (KANB)
# Strain Enhancement terms
Qc = 7.88e4         # activation energy for low trial 6.0e4 for high 11.5e4 for control 7.88e4
A_ = 3.5e-25    # prefactor constant for low trial and control 3.5e-25 for high trial 24.e-25
Strain_n = 3.0      # creep exponent in Glen's law
# Latent heating terms
Latent = 'off' # Top, Both, Full, off
omega = 0.01/const.spy        # added water content during time in crevasse field (sec-1) .0023 for 6C over 15 years
omega_max = 0.05            # limit the water content at 5%
vel_max = 0.2/const.spy     # maximum of the sinusoidal velocity for variable vertical advection experiments

################################################################################################################################

### Define the initial temperature profile ###

Thick = 821. #Thickness of GL11 2B
wavelength=500. # wavelength of foliation

# Use both 2011 inland profiles for initial condition (init)
rawGL11_site2B = np.genfromtxt('./AllTemps_GL11Site2.txt')
Hinit = rawGL11_site2B[:,0]     # height above bed
Tinit = rawGL11_site2B[:,1]     # temperature

# add artificial points at the top and bottom to smooth out the boundary conditions that will be applied
Hinit = np.insert(Hinit, 0, 0.)
Tinit = np.insert(Tinit, 0, -const.rho*const.g*const.gamma*Thick)
Hinit = np.append(Hinit, Thick)
Tinit = np.append(Tinit, (Thick*T_lapse)+T_start)

# use a least squares optimization to get a fourier sine series that approximates the data
def mod(par,z):
    return par[0]*np.sin(z*par[1]+par[2])+par[3]*np.sin(z*par[4]+par[5])+par[6]*np.sin(z*par[7]+par[8])+par[9]
def f(par,z):
    return Tinit - mod(par,z)
par0 = np.array([1,2*pi/1000.,1.,1.,2*pi/1000.,1.,1.,2*pi/1000.,1.,-10.])
Opt = leastsq(f,par0,Hinit)[0]

################################################################################################################################

### Define the mesh, and set up the problem ###

ndim = 2 - 1    # number of dimensions, -1 for python indexing
dt = 1.     # time step
t=0.        # start at time t=0
# Define a time for water input
t_w_start = (46.-30.)/(Uvel/1000.) #29. is the distance from terminus to crevasse field
t_w_end = (46.-28.5)/(Uvel/1000.) #29. is the distance from terminus to crevasse field

# Define the mesh from the ice surface to some max depth and width values, refine the mesh with a larger n
parameters['allow_extrapolation'] = True
#Mesh = IntervalMesh(n,0,Thick)
Mesh = RectangleMesh(Point(0,0),Point(1.5*wavelength,Thick),50,200)
V = FunctionSpace(Mesh,'CG',1)

# Define the initial temperature profile based on the optimization above
T_ = Expression('T0+p0*sin(x[ndim]*p1+p2)+p3*sin(x[ndim]*p4+p5)+p6*sin(x[ndim]*p7+p8)+p9',
                ndim=ndim,T0=const.T0,p0=Opt[0],p1=Opt[1],p2=Opt[2],p3=Opt[3],p4=Opt[4],
                p5=Opt[5],p6=Opt[6],p7=Opt[7],p8=Opt[8],p9=Opt[9])

#Pressure-Dependent Melting Point (degC)
Tm = Expression('T0-rho*g*gamma*(Thick-x[ndim])',ndim=ndim,T0=const.T0,rho=const.rho,g=const.g,gamma=const.gamma,Thick=Thick)

# Define the enthalpy variable
H = Function(V)
H0 = Expression('(temp-Tm)*C',temp=T_,Tm=Tm,C=const.C)
H_1 = interpolate(H0,V)
Hmax = omega_max*const.L

# Temperature for strain heating and output
T = Expression('min((H)/C + Tm,Tm)',H=H_1,C=const.C,Tm=Tm)

### Source Terms###
# The energy released by strain dissipation (Pa sec-1) Cuffey and Paterson (pg 73)
q_strain = Expression('2.*A_*exp(-(Qc/R)*((1./(temp-Tm+T0))-(1./(263.-Tm+T0))))\
               *pow((rho*g*sqrt(pow((Thick-x[ndim]),2.))*slope),(Strain_n+1.))',
                ndim=ndim, A_=A_, Qc=Qc, R=const.R, temp=T, Tm=Tm, T0=const.T0, rho=const.rho,
                g=const.g, Thick=Thick, slope=0., Strain_n=Strain_n)

# latent heating by water in crevasses, only non-zero within crevasse field
if Latent == 'Top':
    # for top only
    q_latent = Expression('x[ndim] > s_crevasse ? rhow*L*omega : 0.',
                ndim=ndim,s_crevasse=Thick-100.,rhow=const.rhow,L=const.L,omega=0.)
elif Latent == 'Both':
    # for both top and bottom
    q_latent = Expression('x[ndim] > s_crevasse ? rhow*L*omega : (x[ndim] < b_crevasse ? rhow*L*omega : 0.)',
                          ndim=ndim,s_crevasse=Thick-100.,b_crevasse=100.,rhow=const.rhow,L=const.L,omega=0.)
else:
    q_latent = Expression('rhow*L*omega',rhow=const.rhow,L=const.L,omega=0.)

################################################################################################################################

### Define the functions that I need, and the Boundaries ###

# update the parameters at the surface boundary
def SurfConstraints(Thick,t,dt):
    # elevation-dependent ablation (m/yr) lapse rate is from van de wal 2012 (~ -4.22 at margin from mean site 4 in van de wal 2012)
    Ablate = (Thick*Abl_lapse)+Abl_start
    # air temperature, lapse rate from Fausto 2009 and from Van As 2012 (~ -5 at Kanger and KAN_B)
    AirT = (Thick*T_lapse)+T_start
    # Slope and thickness updates
    slope = (-Ablate)/(Uvel)
    Thick += Ablate*dt
    return AirT,Thick,slope

# The surface boundary condition will update at each time step because the surface is lowering
def AirBC(AirT,Thick):
    # Define an air subdomain for the boundary condition of air temp from MET station
    class Air(SubDomain):
        def inside(self, x, on_boundary):
            # Anything above the snow surface (lower depth) is 'air'
            return near(x[ndim],Thick) and on_boundary
    # Update the surface boundary condition to the current air temp (include one for enthalpy to get rid of old water)
    bc_air = DirichletBC(V, (AirT)*const.C, Air())
    return bc_air

# Set up a Neumann Boundary at the bed (geothermal heat flux)
class Bottom(SubDomain):
    def inside(self,x,on_boundary):
        return x[ndim] < const.tol and on_boundary
# Instantiate sub-domain instances
bottom = Bottom()
boundaries = FacetFunction("size_t", Mesh, 1)
boundaries.set_all(0)
# mark ds(1) as bed to be used in the variational problem with q_geo
bottom.mark(boundaries, 1)
ds = Measure("ds")[boundaries]

################################################################################################################################

### Set up the variational problem ###

#velocity field for advection term (for variable vertical advection experiments)
vert_vel = Expression('-amp*cos(2.*pi*(3/2.)*(x[0]/dist))',amp=vel_max,pi=np.pi,dist=wavelength*1.5,Thick=Thick)
vel = as_vector([Constant(0.),vert_vel])
# Define a function for the enthalpy diffusivity, this needs to be discontinuous at the phase boundary
kappa = 0.5*(const.k/(const.C*const.rho)-const.nu/const.rho)*(1.-((2./np.pi)*atan(.01*(H))))+const.nu/const.rho

# Define test and trial function
u = TrialFunction(V)
v = TestFunction(V)

# Set up the variational form, (see Brinkerhoff 2013)
F_1 = ((u-H_1)/(const.spy*dt) + dot(vel,nabla_grad(u)))*v*dx + kappa*inner(nabla_grad(u), nabla_grad(v))*dx\
                - 1./(const.rho)*(const.q_geo*v*ds(1) + (q_strain+q_latent)*v*dx)

a = lhs(F_1)
L = rhs(F_1)

# I am going to deform the mesh when I have a lowering surface so I need to grab a few things
Y = Mesh.coordinates()[:,1]
# I need the degree of freedom map for deforming the mesh
v2dof_map = vertex_to_dof_map(V)
dof2v_map = dof_to_vertex_map(V)

#############################################################################################################################

### Loop through the problem until ice is completely melted ###

# Depth array and empty 'out' arrays for data output, high low and mid for variable vertical advection experiments
out_high = np.empty((0,101))
out_low = np.empty((0,101))
out_mid = np.empty((0,101))
depth_out = np.empty((0,101))
# Compute solution while the ice sheet thickness is greater than 0 meters
while Thick > 0.:
    #Update all variables
    LastThick = Thick
    AirT, Thick, slope = SurfConstraints(Thick,t,dt)

    print '', 'Time = ', t
    print 'Thickness = ', Thick
    print 'Slope = ', slope
    Tm.Thick=Thick
    T.Tm=Tm
    q_strain.temp = T
    q_strain.Tm = Tm
    q_strain.Thick=Thick
    q_strain.slope=slope

    # Before I deform the mesh I need to tell each node what its new locations temperature is going to be
    Hold = np.zeros(Mesh.num_vertices())
    for i in range(len(Mesh.coordinates())):
        node = Mesh.coordinates()[i]
        # Grab the temperature from a lower location where the node is going to be moved, Thick/LastThick
        Hold[i] = H_1(node[0],node[1]*Thick/LastThick)
    H_1.vector()[:] = Hold[dof2v_map]

    # bring each node down by the ratio Thick/LastThick
    Y *= Thick/LastThick
    Mesh.bounding_box_tree().build(Mesh)

    # Add water source within time contraints
    if Latent != 'off' and t > t_w_start and t < t_w_end:
        q_latent.s_crevasse = Thick-100.
        q_latent.omega = omega
    else:
        q_latent.omega = 0.

    # update the air boundary condition and solve the diffusion equation for the temperature variable
    bc_air = AirBC(AirT,Thick)
    solve(a==L,H,bc_air)
    # Limit the enthalpy at Hmax
    H.vector()[H.vector()[:]>Hmax] = Hmax

    #plot(H,range_min=const.C*(-15.),range_max=const.C*(1.))
    T.H = H
    H_1.assign(H)

    # export data
    if t%.5 < 1e-5 or t%.5 > .5-1e-5:
        Depths = np.linspace(0,Thick,100)
        depth_out = np.append(depth_out,[np.insert(Depths, 0, t)], axis=0)
        # Export temps in Celsius
        Tout_high = np.array([T(wavelength,d)-const.T0 for d in Depths])
        Tout_high[np.where(Depths>Thick)]=np.nan
        out_high = np.append(out_high,[np.insert(Tout_high, 0, t)], axis=0)
        Tout_low = np.array([T(0.5*wavelength,d)-const.T0 for d in Depths])
        Tout_low[np.where(Depths>Thick)]=np.nan
        out_low = np.append(out_low,[np.insert(Tout_low, 0, t)], axis=0)
        Tout_mid = np.array([T(0.75*wavelength,d)-const.T0 for d in Depths])
        Tout_mid[np.where(Depths>Thick)]=np.nan
        out_mid = np.append(out_mid,[np.insert(Tout_mid, 0, t)], axis=0)
    t += dt # update the time variable

#"""
