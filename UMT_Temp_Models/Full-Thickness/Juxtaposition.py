# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:52:13 2016

@author: ben

This model simulates advection diffusion in a two dimensional domain.
The idea to be tested here is whether differential vertical advection 
(either through bed-parallel flow or through variable strain) forces
the development of horizontal temperature gradients faster than they 
are erased by diffusion.

Advection is treated through mesh displacement so that flow distorts the domain.
"""

import numpy as np
import matplotlib.pyplot as plt
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

################################################################################################################################

### Define the initial temperature profile ###

# 33-km     bed = 311   Surf = 978  Thick = 667
# 46-km     bed = 269   Surf = 1090 Thick = 821
Bed = 311.  # Mean bed elevation
Surf = 978. # Mean surface elevation
Thick = Surf-Bed
wavelength=700.
ndim = 2 - 1    # number of dimensions, -1 for python style indexing
dt = 1.     # time step 
   
# Use all temp profiles for initial condition (init)
# './All_Block.txt' all block site except 14N
# './AllTemps_GL11Site2.txt' all inland profiles
#raw_temps = np.genfromtxt('./AllTemps_GL11Site2.txt')
raw_temps = np.genfromtxt('./All_Block.txt')
H_data = raw_temps[:,0]
T_data = raw_temps[:,1]

# use a least squares optimization to get a fourier sine series that approximates the data
def mod(par,z):
    return par[0]*np.sin(z*par[1]+par[2])+par[3]*np.sin(z*par[4]+par[5])+par[6]*np.sin(z*par[7]+par[8])+par[9]
def f(par,z):
    return T_data - mod(par,z)
par0 = np.array([1,2*pi/1000.,1.,1.,2*pi/1000.,1.,1.,2*pi/1000.,1.,-10.])
Opt = leastsq(f,par0,H_data)[0]

# Define the initial temperature profile based on the optimization above
T_ = Expression('T0+p0*sin(x[ndim]*p1+p2)+p3*sin(x[ndim]*p4+p5)+p6*sin(x[ndim]*p7+p8)+p9',
                ndim=ndim,T0=const.T0,p0=Opt[0],p1=Opt[1],p2=Opt[2],p3=Opt[3],p4=Opt[4],
                p5=Opt[5],p6=Opt[6],p7=Opt[7],p8=Opt[8],p9=Opt[9]) 
  

################################################################################################################################

### Define the mesh ###
    
# Define the mesh from the ice surface to some max depth and width values, refine the mesh with a larger n
parameters['allow_extrapolation'] = True    
#Mesh = IntervalMesh(n,Bed,Surf)
Mesh = RectangleMesh(Point(0,Bed),Point(1.5*wavelength,Surf),50,100)
V = FunctionSpace(Mesh,'CG',2)    

### Initial Temperature Function            
T_1 = interpolate(T_,V)                
T = Function(V)

### Velocity field for advection term

# vertical velocity at the surface is a sine function, but it is either constant through the thickness (const_vel) or variable (var_vel) or a combination
vel_max = 0.5    # vertical velocity amplitude m/yr
const_vel = 1. # full column portion of the velocity (bed-parallel)
var_vel = 0.   # variable portion of the velocity (straining) -- this will create some amount of vertical velocity too
vert_vel = Expression('-(dt*amp*(f1+f2*(x[1]-Bed)/Thick)*cos(2.*pi*(x[0]/dist)))',dt=dt,amp=vel_max,f1=const_vel,f2=var_vel,pi=np.pi,dist=wavelength,Bed=Bed,Thick=Thick)
# Horizontal velocity conserves mass
hor_vel = Expression('(dt*amp*f2/Thick)*dist/(2.*pi*(3/2.))*sin(2.*pi*(x[0]/dist))',dt=dt,amp=vel_max,f2=var_vel,pi=np.pi,dist=wavelength,Bed=Bed,Thick=Thick)

# define velocity as a vector and project it into a vectorfunctionspace for displacing the mesh
vel = as_vector([hor_vel,vert_vel])
Q = VectorFunctionSpace(Mesh,'CG',1)
displacement = project(vel,Q)

# define the location of the bed and surface
Surf_exp = Expression('Surf-t*amp*(f1+f2)*cos(2.*pi*(x[0]/dist))',Surf=Surf,t=0.,amp=vel_max,f1=const_vel,f2=var_vel,pi=np.pi,dist=wavelength)
Bed_exp = Expression('Bed-t*amp*(f1)*cos(2.*pi*(x[0]/dist))',Bed=Bed,t=0.,amp=vel_max,f1=const_vel,pi=np.pi,dist=wavelength)

#Pressure-Dependent Melting Point (degC)
Tm = Expression('T0-rho*g*gamma*(Surf-x[ndim])',ndim=ndim,T0=const.T0,rho=const.rho,g=const.g,gamma=const.gamma,Surf=Surf_exp)
T_conditional = Expression('min(Tm,Temp)',Tm=Tm,Temp=T)
    
################################################################################################################################

### Define the Boundaries ###

# Define an air subdomain for the boundary condition of air temp from MET station
class Air(SubDomain):
    def inside(self, x, on_boundary):
        return x[ndim] > 0.5*Surf and (x[0] > const.tol and x[0] < 1.5*wavelength - 1.) and on_boundary
# Update the surface boundary condition to the current air temp (include one for enthalpy to get rid of old water)
bc_air = DirichletBC(V, const.T0+mod(Opt,Surf), Air())

# Set up a Neumann Boundary at the bed
class Bottom(SubDomain):
    def inside(self,x,on_boundary):
        return x[ndim] < Bed + 0.5*Surf and (x[0] > const.tol and x[0] < 1.5*wavelength - 1.) and on_boundary
bc_bed = DirichletBC(V, Tm, Bottom())                

################################################################################################################################

### Set up the problem and run forward in time ###

# Define a function for the diffusivity
kappa = const.k/(const.C*const.rho)

# Define test and trial function
u = TrialFunction(V)
v = TestFunction(V)   
 
# Set up the variational form
F_1 = (u-T_1)/(const.spy*dt)*v*dx + kappa*inner(nabla_grad(u), nabla_grad(v))*dx

a = lhs(F_1)
L = rhs(F_1)

t = dt
time = 50.
while t<=time:  
    print '', 'Time = ', t   

    # advect the mesh according to the velocity field
    ALE.move(Mesh,displacement)
    Mesh.bounding_box_tree().build(Mesh)
    
    # reset the basal boundary condition
    Surf_exp.t = t
    Tm.Surf = Surf_exp
    bc_bed.set_value(Tm)

    # Solve the problem and limit the temp at the melting point
    solve(a==L,T,[bc_air,bc_bed])
    T_conditional.Temp = T
    T.vector()[:] = interpolate(T_conditional,V).vector()[:]

    plot(T,range_min=const.T0-15.,range_max=const.T0+1.)   
    T_1.assign(T)

    t += dt # update the time variable

###################################################################################################

### Get all profiles that I will need

Bed_max = Bed - t*const_vel*vel_max*cos(2.*pi*0.5)
Bed_min = Bed - t*const_vel*vel_max*cos(2.*pi)
Surf_max = Surf - t*(const_vel+var_vel)*vel_max*cos(2.*pi*0.5)
Surf_min = Surf - t*(const_vel+var_vel)*vel_max*cos(2.*pi)

x = np.linspace(0,wavelength*1.5,100)
y = np.linspace(Bed_min,Surf_max+10.,100)

Bed_exp.t = t
bed_profile = [Bed_exp(x_i) for x_i in x]
surf_profile = [Surf_exp(x_i) for x_i in x]

L_profile = np.arange(Bed_max,Surf_max+0.1,0.5)
R_profile = np.arange(Bed_min,Surf_min+0.1,0.5)
Tleft = np.array([T(.5*wavelength,h)-const.T0 for h in L_profile])
Tcenter = np.array([T(wavelength,h)-const.T0 for h in R_profile])


### Plot velocity field

ax1 = plt.subplot(gs[:2,3])

plt.tick_params(axis='both',which='both',labelleft='off')
X,Y = np.meshgrid(x,y)
vel_mag = [[sign(vert_vel(x_i,y_i))*np.sqrt(hor_vel(x_i,y_i)**2.+vert_vel(x_i,y_i)**2.) for x_i in x] for y_i in y]
plt.contourf(x,y,vel_mag,levels=1.1*np.linspace(-vel_max,vel_max,100),cmap='Blues')#,alpha=0.2)
plt.fill_between(x,Bed_min,Surf_max,color='w',alpha=0.7)

num_vecs = 20
x_vecs = np.linspace(0,wavelength*1.5,num_vecs)
y_vecs = np.linspace(Bed,Surf,num_vecs)
X,Y = np.meshgrid(x_vecs,y_vecs)
V_vel = [[vert_vel(x_i,y_i)*const.spy for x_i in x_vecs] for y_i in y_vecs]
U_vel = [[hor_vel(x_i,y_i)*const.spy for x_i in x_vecs] for y_i in y_vecs]
ax1.quiver(X,Y,U_vel,V_vel,angles='xy',scale_units='xy',lw=1.)
ax1.set_xticks(wavelength*np.arange(0,1.6,.5))

cbar = plt.colorbar(ticks=[-0.5,-0.25,0.,0.25,0.5],label='Velocity (m/yr)')
cbar.set_clim(-0.5,0.5)

ax1.plot(x,bed_profile,'k')
plt.fill_between(x,-100.*np.ones(len(x)),bed_profile,color='w')
plt.fill_between(x,-100.*np.ones(len(x)),bed_profile,color='k',alpha=0.5)

plt.fill_between(x,surf_profile,1200.,color='w')
plt.plot(x,surf_profile,c='k')

plt.xlabel('Horizontal Distance (m)')
ax1.grid('on')
plt.ylim(200,1090)


### Plot Temperature Profiles

ax2 = plt.subplot(gs[:2,2])
plt.tick_params(axis='both',which='both',labelleft='off')
#plt.plot(Tstart,Depths,'k')
plt.plot(Tleft,L_profile,'k--')
plt.plot(Tcenter,R_profile,'k')  
plt.grid()
plt.xlabel(u'Temperature (°C)')
#plt.ylabel('Elevation (m)')
#ax2.set_yticks(np.arange(300,901,100))
plt.ylim(200,1090)

ax1.text(.05,.95,'(d)',transform=ax1.transAxes)
ax2.text(.05,.95,'(c)',transform=ax2.transAxes)

#plt.tight_layout()

#plt.savefig('Foliations_BedParallel.jpg',format='jpg',dpi=300)
#plt.savefig('Foliations_VericalStrain.jpg',format='jpg',dpi=300)
#plt.savefig('Foliations_Combination.jpg',format='jpg',dpi=300)
#"""