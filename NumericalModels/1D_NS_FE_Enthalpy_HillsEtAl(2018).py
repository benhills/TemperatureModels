#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 11:31:28 2016

@author: ben
"""

"""
This model is tests some ideas on near-surface heat transfer in the ablation zone.

The execution of the model is carried out in 'NS_Model_Runs.py'
"""

import numpy as np
from time import sleep
from dolfin import *

### Constants ###
### (Van der Veen 2013) ###
class Constants(object):
    def __init__(self):
        self.g = 9.81                            #Gravity (m/s2)
        self.spy  = 60.*60.*24.*365.24           #Seconds per year
        self.spd  = 60.*60.*24.                  #Seconds per day
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
        # Density of Snow (kg m-3) Unfortunately there is not much data on the density of snow in this area
        # not even in the k-transect (Van de Wal et al., 2005) but 300 kg/m3 is what Lefebre et al. (2003) use at Swiss Camp
        # Hooke et al. (1983) use 360-380
        self.rho_snow = 300.      
        #Heat Capacity of snow is the same as ice (Sakazume and Seki 1978 from fukusako 1990)                    
                            
        # Radiation Extinction Coefficient (m-1)
        # Ice 28 to .2 meters then 1.8 (Paterson (1972), citing Lliboutry (1964-5, Tom. p368))
        # Snow it seems like most authors site somewhere between 10-30 (Mellor (1964), properties of snow, Colbeck (1989), Perovich (2007)) but it is largely dependent on wavelength
        # Wheler and Flowers (2011) site Greuell and Konzelmann (1994) who optimize for ~2.5 in ice and 20 in snow at Swiss Camp
        self.lambda_iceA = 28.
        self.lambda_iceB = 1.8
        self.lambda_snow = 20.
        self.RadDepth=0.2           # depth at which the extinction coefficient changes

# initialize constants class
const = Constants()

################################################################################################################################

### Surface Data ###

# Thermal Diffusivity of Snow
# Input the snow density and heat capacity
def snow_diff(rho_snow,C_snow):
    k_snow = 2.5e-6*rho_snow**2-1.23e-4*rho_snow+0.024     #Thermal Conductivity of snow (Calonne 2011)
    kappa_snow = k_snow/(rho_snow*C_snow)      
    return kappa_snow

# Set up the snow layer depth and ice melt through the year using the input ablation data
# Ablation data comes from the temperature-corrected distance to target (from the MET station)
def split_sonic(Sonic,AirT_in,DOY,IceSurfInit):
    IceSurf = [IceSurfInit]
    SnowDepth = [0.]
    Last = 0.
    for i in range(len(DOY)):
        # Find the distance that the surface has moved since the last timestep        
        dz = Sonic[i] - Last
        Last = Sonic[i]        
        # If the distance is negative it means some melting has occured either snow or ice
        if dz < 0.:
            # Melt an amount of snow equal to all the melting or all the available snow            
            snowmelt = max([dz,-SnowDepth[-1]])
            SnowDepth = np.append(SnowDepth,SnowDepth[-1]+snowmelt)
            # Melt an amount of ice equal to whatever is leftover after snowmelt is removed from dz
            icemelt = dz-snowmelt
            IceSurf = np.append(IceSurf,IceSurf[-1]-icemelt)
        # If the distance dz is positive then it means that snow accumulation has occured
        # In this case the ice surface will remain the same and the snow surface will move up
        else:
            SnowDepth = np.append(SnowDepth,SnowDepth[-1]+dz)
            IceSurf = np.append(IceSurf,IceSurf[-1])  
    # return all but the initialization point
    return IceSurf[1:],SnowDepth[1:]

# Everything that is above the snow surface is labeled as air and set to the input temp
def air_boundary(AirT_in,SnowSurf,ind,V,n,r,Tm):   
    # Define an air subdomain for the boundary condition of air temp from MET station
    class Air(SubDomain):
        def inside(self, x, on_boundary):
            # Anything above the snow surface (lower depth) is 'air' 
            # add in 1/(n**(r+1)) - one cell in the refined mesh - so that there are not meshing errors at the surface
            return x[0] <= SnowSurf + 1./((n)**(r+1)) + const.tol
    # Update the surface boundary condition to the current air temp (in Enthalpy)
    bc_air = DirichletBC(V, Expression('(temp-Tm)*C',temp=AirT_in[ind]+const.T0,Tm=Tm,C=const.C), Air())
    return bc_air

################################################################################################################################

# Build a mesh that is refined near where I am going to add water to cryoconite holes
# n is the number of nodes per meter
# r is the number of refinements near the rotten ice later
def build_mesh(r,DomainDepth,n,IceSurfInit,SnowDepth):
    mesh = IntervalMesh(n*DomainDepth,0,DomainDepth)   
    for refinements in range(r):
        cell_markers = CellFunction("bool",mesh)
        cell_markers.set_all(False)
        for cell in cells(mesh):
            if cell.midpoint().x() < IceSurfInit + 1.:  #cell.midpoint().x() > IceSurfInit - max(SnowDepth) - 0.5 and
                    cell_markers[cell] = True
        mesh = dolfin.refine(mesh, cell_markers) 
    return mesh

################################################################################################################################

# Penetration of Solar Radiation
# The energy added by penetration of solar radiation is negative the derivative of the Beer-Lambert law
def radiation_penetration(q_rad,DSnow,DIceA,DIceB,EnRadSnow,EnRadIceA,EnRadIceB,V,Qsw,IceSurfInit,SnowDepth,ind):
    # only run if there is positive shortwave radiation
    if Qsw[ind]>const.tol:
        # Update the snow/ice surface variables
        DSnow.Surf1 = IceSurfInit - SnowDepth[ind]
        DSnow.Surf2 = IceSurfInit
        DIceA.Surf1 = IceSurfInit
        DIceA.Surf2 = IceSurfInit+const.RadDepth
        DIceB.Surf = IceSurfInit+const.RadDepth
        # Define the profile of energy input by radiation based on the Beer-Lambert law, 
        # as well as depth expressions and extinction coefficients for snow and ice
        Q=Qsw[ind]                   # wavelengths above 800nm are extinguished right at the surface (Greuell and Oerlemans 1989)
        EnRadSnow.Qsw = Q    # Radiation at the snow surface
        EnRadSnow.DSnow = DSnow
        QswA = Q*exp(-const.lambda_snow*SnowDepth[ind])  # Radiation at the ice surface
        EnRadIceA.QswA = QswA        
        EnRadIceA.DIceA = DIceA
        EnRadIceB.QswB = QswA*exp(-const.lambda_iceA*const.RadDepth)  # Radiation at the ice surface
        EnRadIceB.DIceB = DIceB
        # Project the energy from radiation into the function space and then add it to H
        q_rad.S,q_rad.A,q_rad.B = EnRadSnow,EnRadIceA,EnRadIceB
    return q_rad

################################################################################################################################

### Define the iterative diffusion function ###

# dt is the time step
# minyrs and converge are the convergence criterion so that it will run at least minyears # of years
# and it will run until temperature difference (from one year to the next) at the domain depth is less than converge
# Profile_dz and ProfileLoc tell the function where the output profile should come from
def ns_enthalpy(AirT_in,DOY,omega_max,Sonic,Qsw,\
            DomainDepth=25,IceSurfInit=3.,n=2,r=5,Tinit=-10.0,dt=1.0,minyrs=10.,converge=1e-2/const.C,Profile_dz=0.1,\
            Kconstant=False,TwoYears=False,Qfactor=0.64,Sfactor=1.,Lsource=0.,Ldepth=0.):

    # Find the snow depths and Ice surface throughout the year
    IceSurf,SnowDepth = split_sonic(Sonic,AirT_in,DOY,IceSurfInit) 
    SnowDepth = SnowDepth*Sfactor
    SnowSurf = IceSurfInit
    # add one point on to the end of IceSurf so that the last velocity measurment is zero
    IceSurf = np.append(IceSurf,IceSurf[-1])

    # create a velocity expression that will be set equal to the ablation rate
    vel = Expression('abl',abl=0.)
    # Define the lithostatic pressure
    P = Expression('rho*g*max(0.,surf-x[0])',rho=const.rho, g=const.g, surf=IceSurfInit)
    # Define temperature at the melting point
    Tm = Expression('T0-gamma*P',T0=const.T0, gamma=const.gamma, P=P)    

    # Define all the functions that I need for adding energy from penetration of radiation
    # Depths are ~inf outside of the material and are depth minus surface inside    
    DSnow = Expression('x[0] < Surf1 ? 100000. : (x[0] > Surf2 ? 100000. : max(0.,x[0]-Surf1))',Surf1=0.,Surf2=0.)
    DIceA = Expression('x[0] < Surf1 ? 100000. : (x[0] > Surf2 ? 100000. : max(0.,x[0]-Surf1))',Surf1=0.,Surf2=0.)
    DIceB = Expression('x[0] < Surf ? 100000. : max(0.,x[0]-Surf)',Surf=0.)
    # Energy from radiation (J/kg) is negative the derivative of the Beer-Lambert law times the time step    
    EnRadSnow = Expression('lambda_snow*Qsw*exp(-lambda_snow*DSnow)',
                           rho=const.rho,Qsw=0.,lambda_snow=const.lambda_snow,DSnow=DSnow)
    EnRadIceA = Expression('lambda_iceA*QswA*exp(-lambda_iceA*DIceA)',
                          rho=const.rho,QswA=0.,lambda_iceA=const.lambda_iceA,DIceA=DIceA)
    EnRadIceB = Expression('lambda_iceB*QswB*exp(-lambda_iceB*DIceB)',
                          rho=const.rho,QswB=0.,lambda_iceB=const.lambda_iceB,DIceB=DIceB)
    # Total energy input by radiation
    q_rad = Expression('S+A+B', S=EnRadSnow, A=EnRadIceA, B=EnRadIceB)
    # source function for refreezing below the surface 
    # Ldepth is the penetration limit and Lsource is the water volume freezing every year
    if Lsource > 0.:
        source = const.rhow*Lsource*(const.L)/const.spy
        refreeze = Expression('x[0] > Ldepth + IceSurfInit ? 0. : (x[0] < IceSurfInit ? 0. : source/Ldepth)',
                              Ldepth=Ldepth, IceSurfInit=IceSurfInit,source=source)
    else:
        refreeze = Constant(0)

    # Define the mesh from the ice surface to some max depth and width values, refine the mesh with a larger n
    mesh = build_mesh(r,DomainDepth,n,IceSurfInit,SnowDepth)
    meshx = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh,'CG',1) 
    # Define the enthalpy variable
    H = Function(V)
    H0 = Expression('(temp-Tm)*C',temp=const.T0+Tinit,Tm=Tm,C=const.C)
    H_1 = interpolate(H0,V)
    
    print 'test'    
    
    # Calculate a vector for diffusivities at all points in the mesh
    if Kconstant == True:
        K = const.k/(const.C*const.rho)
    else:
        # snow_diff gives the diffusivity of snow (m2 sec-1)
        kappa_snow = snow_diff(const.rho_snow,const.C)
        # Define a function for the enthalpy diffusivity
        # This uses an arctan as a type of conditional to select between k/(C*rho) and nu/rho
        kappa_ice = 0.5*(const.k/(const.C*const.rho)-const.nu/const.rho)*(1.-((2./np.pi)*atan(.01*(H))))+const.nu/const.rho
        # Incorporate snow, cold ice, and temperate ice into one diffusivity term
        K = 0.5*(kappa_snow-kappa_ice)*(-1.-((2./np.pi)*atan(100.*(meshx[0]-SnowSurf+const.tol))))+kappa_snow    
    
    # Define test and trial function
    u = TrialFunction(V)
    v = TestFunction(V)     
    # Set up the variational form, (see Brinkerhoff 2013)
    F_1 = ((u-H_1)/(const.spd*dt))*v*dx + K*inner(nabla_grad(u), nabla_grad(v))*dx\
                    - dot(vel,u.dx(0))*v*dx - (1/const.rho)*(q_rad+refreeze)*v*dx
    #F_1 = ((u-H_1)/(const.spd*dt))*v*dx + (const.k/(const.rho*const.C))*inner(nabla_grad(u), nabla_grad(v))*dx - dot(vel,u.dx(0))*v*dx - q_rad*v*dx    
    # Set upper and lower limits to the enthalpy variable, upper at max water content, lower at absolute zero
    constraint_Hmax = Expression('(temp-Tm)*C+omega_max*L',temp=const.T0,Tm=Tm,C=const.C,omega_max=omega_max,L=const.L)
    Hmax = interpolate(constraint_Hmax, V)
    constraint_Hmin = Expression('(temp-Tm)*C',temp=0.,Tm=Tm,C=const.C)
    Hmin = interpolate(constraint_Hmin, V)
    # Define the solver parameters
    snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "lu",
                                          "maximum_iterations": 100,
                                          'line_search': 'basic',
                                          "report": True,
                                          "error_on_nonconvergence": True, 
                                          "relative_tolerance" : 1e-11,
                                          "absolute_tolerance" : 1e-7}}
                                          
    # Iteration variables
    yrs = 0.    #number of years run
    diff = 0.   # % difference from last annual enthalpy value
    Hyr = H_1(DomainDepth) # convergence Enthalpy
    spinupStart = 365.        # July 18 2015
    spinupEnd = 365*2+1.     # July 18 2016 add 1 for leap year
    # loop the second year as a spinup (more typical year)
    ts = np.arange(DOY[spinupStart],DOY[spinupEnd]+dt,dt) 
    
    # Depth profile for model output
    SensorD = IceSurfInit + np.arange(0,21+Profile_dz,Profile_dz)
    out = [np.insert(SensorD-IceSurfInit, 0, 99.99)]
    # Last year for determining if it is the last year in 2 year model run and another model output array
    LastYear = 0.
    TwoYearOut = [np.insert(SensorD-IceSurfInit, 0, 99.99)]
    # Temperature expression for exporting temperatures
    if Kconstant == True:
        T = Expression('(H)/C + Tm',H=H,C=const.C,Tm=Tm)
    else:        
        T = Expression('min((H)/C + Tm,Tm)',H=H,C=const.C,Tm=Tm)
    
    # Compute solution over spinup period and one more year if desired
    while yrs <= minyrs or diff > converge or (LastYear<1.5 and TwoYears == True):
        # Recalculate all the convergence parameters
        yrs += 1.
        diff = abs((Hyr - H(DomainDepth))/H(DomainDepth))
        Hyr = H(DomainDepth)
    
        for t in ts:
            ind = np.where(DOY==int(t))[0][0]
            print "Years so far = ", yrs
            print "Day of Year 2014 = ", t
            print "Convergence = ", diff
            print "Ablation = ", IceSurf[ind]
            print "Snow = ", SnowDepth[ind]
            print "Radiation = ", Qsw[ind]
    
            # update expressions that depend ont the ice surface            
            vel.abl = (IceSurf[ind+1] - IceSurf[ind])/const.spd
            SnowSurf = IceSurfInit - SnowDepth[ind]
            # Find the boundary condition for the air(from current snow depth)            
            bc_air = air_boundary(AirT_in,SnowSurf,ind,V,n,r,Tm)
            # Update the radiation source            
            q_rad = radiation_penetration(q_rad,DSnow,DIceA,DIceB,EnRadSnow,EnRadIceA,EnRadIceB,V,Qfactor*Qsw,IceSurfInit,SnowDepth,ind)                
        
            # change F_1 to linear form
            F = action(F_1,H)
            # Compute Jacobian of F
            J = derivative(F, H, u)
            # Set up the non-linear problem
            problem = NonlinearVariationalProblem(F, H, bc_air, J=J)
            problem.set_bounds(Hmin,Hmax)
            # Set up the non-linear solver
            solver = NonlinearVariationalSolver(problem)
            solver.parameters.update(snes_solver_parameters)
            info(solver.parameters, True)
            # Solve the problem
            (iter, converged) = solver.solve()            
            
            #plot(H,range_min=const.C*(-15.),range_max=const.C*(1.))   
            H_1.assign(H)
            sleep(.000001)
            #print min(K)

            # Export data
            if yrs > minyrs and diff < converge:
                # Save the current temp profile to an array (depth profile are defined in the function)
                if abs(t%1) < const.tol or abs(t%1-1) < const.tol:
                    # update temperature expression
                    T.H = H
                    Tout = [T(d)-const.T0 for d in SensorD]
                    if LastYear == 0.:
                        out = np.append(out,[np.insert(Tout, 0, t)], axis=0)        
                    elif LastYear == 1.:
                        TwoYearOut = np.append(TwoYearOut,[np.insert(Tout, 0, t)], axis=0)        




        if yrs > minyrs and diff < converge and TwoYears == True:
            # The last run should be through two full years (length of collected data)
            ts = np.arange(min(DOY),max(DOY)+dt,dt) 
            # update the last year so that it runs one last time
            LastYear += 1.

    # return daily temperatures for each point in the desired profile
    # this should return more than two full years of temperatures
    return out, TwoYearOut

#"""
