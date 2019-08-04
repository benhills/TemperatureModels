#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 17:50:13 2016

@author: ben
"""

"""
This script is meant to run NS_HeatTransfer.py for simulations which test heat
transfer processes at the surface boundary of an ice sheet including:
1) pure conduction of the air temperature 
2) the limit of skin temperature at the melting point
3) ablation which forces a moving surface boundary
4) thermal insulation by a winter snowpack
5) the penetration of shortwave radiation
6) subsurface refreezing of meltwater


This script outputs values into a .txt file that can then be
plotted using PlotSurf.py as well as PlotContour.py. See the RunPlots.py
file to plot a selection of .txt files all at once.

# Model output gives a matrix of temperatures with column one being day of year, row one being the depths of output temperatures
# and all other values being temperatures in degC
# Note that when a simulated sensor melts out, it will then track the ice surface and get buried by snow in the winter instead of 
# remaining fixed in place.
"""

import numpy as np
import pickle
from NS_HeatTransfer import *

### Import the mean daily values to use as a reference
with open('../2014_Data/METdata/MeanDailyMeasurements.pickle', 'rb') as handle:
  MET = pickle.load(handle)
  
AirT = MET['AirT_14']       # Mean Daily Air Temperature degC
Abl = MET['Abl_14']         # Mean Daily Sonic Data (ablation and wintertime accumulation) measured in meters below the initial point
DOY = MET['DOY_14']         # Day of Year

# Import Radiation data from PROMICE, KAN_L which is SW of our site
Kdata = np.genfromtxt('../2014_Data/METdata/KAN_L_day.txt',skip_header=1)
KSW_in = Kdata[2146:2904,13]    # July 18 2014 to August 14 2016
KSW_out = Kdata[2146:2904,15]
KSW = KSW_in - KSW_out

##############################################################################

### Below is the model definition to use as a reference for inputting variables

#def ns_enthalpy(AirT_in,DOY,omega_max,Sonic,Qsw,\
#            DomainDepth=25,IceSurfInit=3.,n=2,r=5,Tinit=0.0,\
#            dt=1.0,minyrs=5.,converge=1e-5,Profile_dz=0.1,Kconstant=False,TwoYears=False,Sfactor=1.):


# Define the parameters for the initial run 
# No water so mwe = 0.0 and CryoDepth is small so that the refinement of the mesh is small
# The ice surface is taken from the ablation data but we need to get rid of the winter snow so use the SurfMassBal function
omega_max = 0.5
IceSurf,Trash = split_sonic(Abl,AirT,DOY,0.)

# Simple diffusion of the air temperature
Diff,trash = ns_enthalpy(AirT,DOY,omega_max,np.zeros_like(Abl),np.zeros_like(KSW),Kconstant=True)
np.savetxt('Results_Figures/SimpleDiff.txt',Diff)

#Melting Point
AirT[AirT>0.]=0.
omega_max=0.
Melt,trash = ns_enthalpy(AirT,DOY,omega_max,np.zeros_like(Abl),np.zeros_like(KSW))
np.savetxt('Results_Figures/Melt.txt',Melt)

#Lowering Ice Surface
Ablate,trash = ns_enthalpy(AirT,DOY,omega_max,-IceSurf,np.zeros_like(KSW))
np.savetxt('Results_Figures/Ablate.txt',Ablate)

# Winter accumulation
Snow,trash = ns_enthalpy(AirT,DOY,omega_max,Abl,np.zeros_like(KSW))
np.savetxt('Results_Figures/Snow.txt',Snow)

# Run with radiation and with water allowed to accumulate
omega_max = 0.5
RadW,TwoYearRad = ns_enthalpy(AirT,DOY,omega_max,Abl,KSW,TwoYears=True)
np.savetxt('Results_Figures/Radiation.txt', RadW)
np.savetxt('Results_Figures/RadiationTwoYears.txt', TwoYearRad)

# Two runs with an enhanced snowpack
Snow2,TwoYearSnow2 = ns_enthalpy(AirT,DOY,omega_max,Abl,KSW,Sfactor=2.,TwoYears=True)
np.savetxt('Results_Figures/Snow2xTwoYears.txt',TwoYearSnow2)
Snow3,TwoYearSnow3 = ns_enthalpy(AirT,DOY,omega_max,Abl,KSW,Sfactor=3.,TwoYears=True)
np.savetxt('Results_Figures/Snow3xTwoYears.txt',TwoYearSnow3)

#AirT[AirT>0.]=0.

# 20 runs with water input for refreezing
for d in [10.]:#np.arange(0.5,10.5,.5):
    Water,WaterTwoYears = ns_enthalpy(AirT,DOY,omega_max,Abl,KSW,Ldepth=d,Lsource=0.05,TwoYears=True)
    np.savetxt('Results_Figures/WaterTwoYears_%s.txt'%d,WaterTwoYears)
    np.savetxt('Results_Figures/Water_%s.txt'%d,Water)
    Water,WaterTwoYears = ns_enthalpy(AirT,DOY,omega_max,Abl,KSW,Ldepth=d,Lsource=0.05,TwoYears=True,Sfactor=2.)
    np.savetxt('Results_Figures/Source_Snow2X_%s.txt'%d,WaterTwoYears)
#"""