# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 18:17:36 2022
@author: Juan de la Peña Toledo
"""

import math
import numpy as np
import matplotlib.pyplot as plt

#FUNCTION TO BE USED BY OTHER FILES. RETURNS THE ESTIMATED INDUCTANCE FOR A GIVEN FLUX
def estimateInductance(V_max, Lm, L_A, f, K, N, flux):
    # ***** Repetition from above *****
    # V_max is referred to rms value across winding 
    # Variables required to build the saturation curve
    # flux_linkage_M = V_max/(2*math.pi*f) #Wb previous expression #this is used if V_max is entered in peak value (instead of rms)
    flux = abs(flux)
    flux_linkage_M = V_max/((2*math.pi/math.sqrt(2))*f) #Wb 2*math.pi/math.sqrt(2) = 4.442882938158366
    I_M = flux_linkage_M/Lm #A
    flux_linkage_K = K * flux_linkage_M
    A = L_A/(flux_linkage_K**2)
    B = (L_A*I_M - flux_linkage_M)/flux_linkage_K
    C = I_M*(L_A*I_M - flux_linkage_M + flux_linkage_K)
    D = (-B-math.sqrt(B**2-4*A*C))/(2*A)
    
    # Create a piecewise linear Inductance vector
    last_point_flux = flux_linkage_K*1.05
    first_point_flux = flux_linkage_M
    inductances = np.empty(N+1)
    fluxes = np.linspace(first_point_flux, last_point_flux, N)
    currents = np.empty(N)
    prev_flux = 0
    prev_current = 0
    for point in range(N):
        flux_linkage = fluxes[point]
        currents[point] = (math.sqrt((flux_linkage - flux_linkage_K)**2 + 4*D*L_A) + flux_linkage - flux_linkage_K)/(2*L_A) - D/flux_linkage_K
        inductances[point] = (flux_linkage - prev_flux)/(currents[point] - prev_current)
        prev_flux = flux_linkage
        prev_current = currents[point]
        
    inductances[0] = Lm #the calculated value is very similar, but not exact.
    inductances[N] = L_A #add air core inductance as the last entry of the array.
    
    # At this point we have:
    # 1. fluxes array: N entries.
    # 2. currents array: N entries. Just the current value for the flux at that same entry.
    # 3. inductances array: N+1 entries.
    # 4. fluxes and inductances arrays are related in the following way:
    
    # For flux > fluxes[N-1], use inductances [N]
    # For flux <= fluxes[0], use inductances[0]
    # For fluxes[i] < flux <= fluxes[i+1], use inductances[i+1]
    
    #
    #  i+1        fluxes[i]          inductances[i]
    #   1      flux_linkage_M             Lm
    #   ·            ·                    ·
    #   ·            ·                    · 
    #   N            K                close to L_A
    #  N+1          None                  L_A
    #
    
    # ***** NEW CODE *****
    if flux <= fluxes[0]:
        return inductances[0]
    elif flux > fluxes[N-1]:
        return inductances[N]
    else:
        for i in range(N-1):
            if fluxes[i] < flux and flux <= fluxes[i+1]:
                return inductances[i+1]
                break
                
def printSaturationCurve(V_max, Lm, L_A, f, K, N):
    # Variables required to build the saturation curve
    flux_linkage_M = V_max/((2*math.pi/math.sqrt(2))*f) #Wb previous expression
    #flux_linkage_M = V_max/(4.442882938158366*f) #Wb
    I_M = flux_linkage_M/Lm #A
    flux_linkage_K = K * flux_linkage_M
    A = L_A/(flux_linkage_K**2)
    B = (L_A*I_M - flux_linkage_M)/flux_linkage_K
    C = I_M*(L_A*I_M - flux_linkage_M + flux_linkage_K)
    D = (-B-math.sqrt(B**2-4*A*C))/(2*A)

    # Create a piecewise linear Inductance vector
    #last_point_flux = flux_linkage_K
    last_point_flux = flux_linkage_K*1.05 #1.05 is used to go beyod the knee point and have a more precise linearized curve
    first_point_flux = flux_linkage_M
    inductances = np.empty(N+1)
    fluxes = np.linspace(first_point_flux, last_point_flux, N)
    currents = np.empty(N)
    prev_flux = 0
    prev_current = 0
    for point in range(N):
        flux_linkage = fluxes[point]
        currents[point] = (math.sqrt((flux_linkage - flux_linkage_K)**2 + 4*D*L_A) + flux_linkage - flux_linkage_K)/(2*L_A) - D/flux_linkage_K
        inductances[point] = (flux_linkage - prev_flux)/(currents[point] - prev_current)
        prev_flux = flux_linkage
        prev_current = currents[point]
        
    inductances[0] = Lm #the calculated value is very similar, but not exact.
    inductances[N] = L_A #add air core inductance as the last entry of the array.

    # At this point we have:
    # 1. fluxes array: N entries.
    # 2. currents array: N entries. Just the current value for the flux at that same entry.
    # 3. inductances array: N+1 entries.
    # 4. fluxes and inductances arrays are related in the following way:
    #
    #  i+1        fluxes[i]          inductances[i]
    #   1      flux_linkage_M             Lm
    #   ·            ·                    ·
    #   ·            ·                    · 
    #   N     flux_linkage_K*Z        close to L_A
    #  N+1          None                  L_A
    #
    # For flux > K, use L_A as inductance OR For flux > fluxes[N-1], use inductances [N]
    # For flux <= flux_linkage_M, use Lm as inductance OR For flux <= fluxes[0], use inductances[0]
    # For fluxes[i] < flux <= fluxes[i+1], use inductances[i+1]

    #GRAPHING
    I_list = list()
    flux_list = list()
    fig = plt.figure(dpi=250, constrained_layout=True)
    ax1 = fig.add_subplot()
    fig = plt.figure(dpi=250, constrained_layout=True)
    ax2 = fig.add_subplot()

    flux_stop = flux_linkage_M*(0.1+1.05*K) #give some margin with + 0.1
    flux=0
    dflux = 0.1
    while flux <= flux_stop:
        flux_linkage = flux
        i_m = (math.sqrt((flux_linkage - flux_linkage_K)**2 + 4*D*L_A) + flux_linkage - flux_linkage_K)/(2*L_A) - D/flux_linkage_K
        flux_list.append((flux_linkage))
        I_list.append(i_m)
        flux = flux + dflux

    # For a better plot
    currents = np.insert(currents, 0, 0)
    currents = np.append(currents, i_m)
    fluxes = np.insert(fluxes, 0, 0)
    fluxes = np.append(fluxes, flux_stop)

    #real values graph
    ax1.plot(I_list, flux_list, linewidth=3)
    ax1.plot(currents, fluxes, 'o--')
    ax1.set_xlabel("Magnetizing current [A]")
    ax1.set_ylabel("Flux-linkage [Wb]")
    ax1.set_title("Saturation curve (seen from HV winding)", pad=15)
    try:
        ax1.figure.savefig(r'results/saturation_curve_hv.png')
    except:
        print("Error saving image file")
    
    #per unit graph
    I_list_pu = [item/I_M for item in I_list]
    flux_list_pu = [item/flux_linkage_M for item in flux_list]
    currents_pu = [item/I_M for item in currents]
    fluxes_pu = [item/flux_linkage_M for item in fluxes]
    ax2.plot(I_list_pu, flux_list_pu, linewidth=3)
    ax2.plot(currents_pu, fluxes_pu, 'o--')
    ax2.set_xlabel("Magnetizing current [pu]")
    ax2.set_ylabel("Flux-linkage [pu]")
    ax2.set_title("Saturation curve", pad=15)
    try:
        ax2.figure.savefig(r'results/saturation_curve_pu.png')
    except:
        print("Error saving image file")
    
if __name__ == "__main__":
    #test 
    
    V_max = 11000/math.sqrt(3) #V Vphase
    #V_max = 11000
    Lm = 61.1357 #H
    L_A = 0.07336 #H
    f = 50 #Hz
    K = 1.15
    N = 10
    printSaturationCurve(V_max, Lm, L_A, f, K, N)