# -*- coding: utf-8 -*-
"""
Created on April 6 2022
@author: Juan de la Peña

"""

# 3Φ. Wye/g-Delta connection
# Assumptions:
# 1. Symmetric conditions, except for Lm ~
# 2. R, L, Lm, Rm, and Znetowork should be indicated per phase
# 3. Neutral connected to ground
#
# ΦA
#                    V_AN_bus
# V_AN --- Znetwork --- / --- R1 ----L1-------------------> I_delta
#                                          |       |
#                                         Lm ~     Rm
#                                          |       |   
#                                          N       N
# 
# Internal voltages always refered to neutral: Vm_A => Vm of phase A refered to neutral
# V_AN_bus phase bus voltage
# V_AB_bus line bus voltage
#
# IMPORTANT --> VALID SCRIPT ONLY IF (L*Lm/Rm) IS SIGNIFICANTLY SMALL -->
# REDUCES ORDER OF DIFFERENTIAL EQUATION

import math
import numpy as np
from Saturation_curve_V2 import estimateInductance
from Saturation_curve_V2 import printSaturationCurve
import matplotlib.pyplot as plt

def EnergizationWyegDelta(V_L, V2_L, L1, L2, Lm, L_A, R1, R2, Rm, phi_0, f, flux_linkage_A_0, flux_linkage_B_0, flux_linkage_C_0, Rnetwork, Lnetwork, K, N, dt, t_stop, change_resolution, bus_energ_voltage_pu, V_source):
    """
    Function to perform the energization simulation of a Wye - Delta transformerm with primary winding grounded
    
    Winding 1 is energized, secondary is unloaded. Winding 1 should be the HV side, as it is energized from the grid.
    
    Description of INPUT parameters: 
    V_L --> Line rated voltage of winding 1, in rms. 
    V_2L --> Line rated voltage of winding 2, in rms. Converted AFTERWARDS to LINE for Wye and PHASE for Delta.
    L1 --> Leakage inductance winding 1, in H
    L2 --> Leakage inductance winding 2, in H
    Lm --> Unsaturated magnetizing inductance, with respect winding 1, in H
    L_A --> Saturated magnetizing inductance, with respect winding 1, in H
    R1 --> Winding 1 resistance, in ohm
    R2 --> Winding 2 resistance, in ohm
    Rm --> Magnetizing resistance, with respect winding 1, in ohm
    phi_0 --> Phase shift voltage source with respect to reference. Line voltage BC is at 0 degrees if no shift is applied.
    f --> Frequency, in Hz
    flux_linkage_A/B/C_0 --> Residual flux of each phase, in Wb
    Rnetwork & Lnetwork --> Equivalent resistance and inductance of connecting bus
    K --> Knee point, in pu
    N --> Number of points for linearization of saturation curve. 10 is recommended.
    dt --> Time step size, in s. 5*10**-6 is recommended.
    t_stop --> Simulation time, in s. Recommended up to 0.2 - 0.5 s
    change_resolution --> Crops the original solutions in a change_resolution amount to be more efficient in a later stage. Should be a number >= 1.
    bus_energ_voltage_pu --> Voltage in pu of energizing bus
    V_source --> RMS base voltage of source in V
    
    Description of OUTPUT parameters, all of them cropped with change_resolution:
    t_virtual_list --> time, in s.    
    I_A_virtual_list, I_B_virtual_list, I_C_virtual_list --> Phase current, in A
    V_AB_bus_rms_virtual_list, V_BC_bus_rms_virtual_list, V_CA_bus_rms_virtual_list --> Line rms voltage of connecting bus, in V
    freqs_scaling_red, I_A_list_fft_mag_scaling_red, I_B_list_fft_mag_scaling_red --> For harmonic content of current waveform. Only phases A and B because phase C is almost identical to B
    V_AN_bus_rms_virtual_list, V_BN_bus_rms_virtual_list, V_CN_bus_rms_virtual_list, I_A_rms_virtual_list, I_B_rms_virtual_list, I_C_rms_virtual_list --> Sliding RMS arrays
    
    """
    
    # Calculate angular speed
    w = 2*math.pi*f
    
    # Include Rnetwork and Lnetwork inside R and L, as this will not affect the inrush current calculation methodology
    R = R1 + Rnetwork
    L = L1 + Lnetwork
    
    # Create variables for three phase system
    V_F = V_L/math.sqrt(3) #V #tension de fase rms
    V_L_max = V_L*math.sqrt(2) #V #tension de linea pico
    V_F_max = V_F*math.sqrt(2) #V #tension de fase pico
    
    # Secondary resistance and inductance refered to the first winding
    L21 = 3*L2*(V_F/V2_L)**2 #Leakage reactance of secondary winding referred to first winding: it should have the same value as L1
    R21 = 3*R2*(V_F/V2_L)**2 #Winding resistance of secondary winding referred to first winding: it should have the same value as R1
    
    #NEEDED FOR GRAPHS
    fig1 = plt.figure(dpi=250, constrained_layout=True)
    ax1 = fig1.add_subplot()
    fig2 = plt.figure(dpi=250, constrained_layout=True)
    ax2 = fig2.add_subplot()
    fig3 = plt.figure(dpi=250, constrained_layout=True)
    ax3 = fig3.add_subplot()
    fig4 = plt.figure(dpi=250, constrained_layout=True)
    ax4 = fig4.add_subplot()
    fig5 = plt.figure(dpi=250, constrained_layout=True)
    ax5 = fig5.add_subplot()
    fig6 = plt.figure(dpi=250, constrained_layout=True)
    ax6 = fig6.add_subplot()
    fig7 = plt.figure(dpi=250, constrained_layout=True)
    ax7 = fig7.add_subplot()
    fig8 = plt.figure(dpi=250, constrained_layout=True)
    ax8 = fig8.add_subplot()
    
    #Create lists
    I_A_list = []
    I_B_list = []
    I_C_list = []
    I_A_rms_list = []
    I_B_rms_list = []
    I_C_rms_list = []
    I_0_list = []
    t_list = []
    flux_linkage_A_list = []
    flux_linkage_B_list = []
    flux_linkage_C_list = []
    V_AN_bus_list = []
    V_BN_bus_list = []
    V_CN_bus_list = []
    V_AB_bus_rms_list = []
    V_AN_bus_rms_list = []
    V_BC_bus_rms_list = []
    V_BN_bus_rms_list = []
    V_CA_bus_rms_list = []
    V_CN_bus_rms_list = []
    V_AB_list = []
    V_BC_list = []
    V_CA_list = []
    V_AN_list = []
    V_BN_list = []
    V_CN_list = []
    I_delta_list = []
    
    #Initialization
    Il_A = 0 #initial conditions
    Il_B = 0 #initial conditions
    Il_C = 0 #initial conditions
    I_A = 0 #initial conditions
    I_B = 0 #initial conditions
    I_C = 0 #initial conditions
    I_delta = 0 #initial conditions
    t = 0 #initial value
    flux_linkage_A = flux_linkage_A_0 #initial value
    flux_linkage_B = flux_linkage_B_0 #initial value
    flux_linkage_C = flux_linkage_C_0 #initial value
    V_AB_bus_rms = V_L #initial value
    V_AN_bus_rms = V_F #initial value
    V_BC_bus_rms = V_L #initial value
    V_BN_bus_rms = V_F #initial value
    V_CA_bus_rms = V_L #initial value
    V_CN_bus_rms = V_F #initial value
    I_A_rms = 0 #initial value
    I_B_rms = 0 #initial value
    I_C_rms = 0 #initial value
    
    #trapezoidal variables
    flux_linkage_sum_prev = 0 
    I_delta_prev = I_delta
    I_delta_prev_prev = I_delta_prev 
    Il_A_prev = 0
    V_AN_prev = V_source*math.sqrt(2)/math.sqrt(3)*V_source*math.sqrt(2)/math.sqrt(3)*math.sin(phi_0 + 90*2*math.pi/360)*bus_energ_voltage_pu
    flux_linkage_A_prev = flux_linkage_A_0 #initial value
    Il_B_prev = 0
    V_BN_prev = V_source*math.sqrt(2)/math.sqrt(3)*math.sin(phi_0 - 30*2*math.pi/360)*bus_energ_voltage_pu
    flux_linkage_B_prev = flux_linkage_B_0 #initial value
    Il_C_prev = 0
    V_CN_prev = V_source*math.sqrt(2)/math.sqrt(3)*math.sin(phi_0 - 150*2*math.pi/360)*bus_energ_voltage_pu
    flux_linkage_C_prev = flux_linkage_C_0 #initial value
    
    while t <= t_stop:
        
        #TENSIONES DE LINEA
        V_AB = V_source*math.sqrt(2)*math.sin(w*t + phi_0 + 120*2*math.pi/360)*bus_energ_voltage_pu #line voltage A-B --> + 120º
        V_BC = V_source*math.sqrt(2)*math.sin(w*t + phi_0)*bus_energ_voltage_pu #line voltage B-C --> 0º
        V_CA = V_source*math.sqrt(2)*math.sin(w*t + phi_0 - 120*2*math.pi/360)*bus_energ_voltage_pu #line voltage C-A --> - 120º
        
        #TENSIONES DE FASE
        V_AN = V_source*math.sqrt(2)/math.sqrt(3)*math.sin(w*t + phi_0 + 90*2*math.pi/360)*bus_energ_voltage_pu #phase voltage A-N --> + 90º
        V_BN = V_source*math.sqrt(2)/math.sqrt(3)*math.sin(w*t + phi_0 - 30*2*math.pi/360)*bus_energ_voltage_pu #phase voltage B-N --> - 30º
        V_CN = V_source*math.sqrt(2)/math.sqrt(3)*math.sin(w*t + phi_0 - 150*2*math.pi/360)*bus_energ_voltage_pu #phase voltage C-N --> - 150º
        
        #calculate variable magnetizing inductance
        Lm_A_variable = estimateInductance(V_F, Lm, L_A, f, K, N, abs(flux_linkage_A)) #don't need to introduce the abs value as this is already done in the function
        Lm_B_variable = estimateInductance(V_F, Lm, L_A, f, K, N, abs(flux_linkage_B))
        Lm_C_variable = estimateInductance(V_F, Lm, L_A, f, K, N, abs(flux_linkage_C))
        
        #---- FIRST ESTIMATES ----
        #main ΦA 
        B_A = Lm_A_variable*R/Rm + L + Lm_A_variable
        Il_A = 1/(1+dt/2*R/B_A)*(Il_A_prev + dt/2*((V_AN_prev - I_delta_prev*R - (I_delta_prev - I_delta_prev_prev)/dt*L)/B_A - Il_A_prev*R/B_A + (V_AN - I_delta*R - (I_delta - I_delta_prev)/dt*L)/B_A))
        I_A = Il_A + Lm_A_variable/Rm*(Il_A - Il_A_prev)/dt + I_delta 
        flux_linkage_A = flux_linkage_A_prev + Lm_A_variable*(Il_A - Il_A_prev)
        
        #main ΦB 
        B_B = Lm_B_variable*R/Rm + L + Lm_B_variable
        Il_B = 1/(1+dt/2*R/B_B)*(Il_B_prev + dt/2*((V_BN_prev - I_delta_prev*R - (I_delta_prev - I_delta_prev_prev)/dt*L)/B_B - Il_B_prev*R/B_B + (V_BN - I_delta*R - (I_delta - I_delta_prev)/dt*L)/B_B))
        I_B = Il_B + Lm_B_variable/Rm*(Il_B - Il_B_prev)/dt + I_delta
        flux_linkage_B = flux_linkage_B_prev + Lm_B_variable*(Il_B - Il_B_prev)
        
        #main ΦC 
        B_C = Lm_C_variable*R/Rm + L + Lm_C_variable
        Il_C = 1/(1+dt/2*R/B_C)*(Il_C_prev + dt/2*((V_CN_prev - I_delta_prev*R - (I_delta_prev - I_delta_prev_prev)/dt*L)/B_C - Il_C_prev*R/B_C + (V_CN - I_delta*R - (I_delta - I_delta_prev)/dt*L)/B_C))
        I_C = Il_C + Lm_C_variable/Rm*(Il_C - Il_C_prev)/dt + I_delta
        flux_linkage_C = flux_linkage_C_prev + Lm_C_variable*(Il_C - Il_C_prev)
        
        #corriente en el bucle delta --> calculo mediente método trapezoidal y flujos
        flux_linkage_sum = flux_linkage_A + flux_linkage_B + flux_linkage_C
        I_delta_prev_prev = I_delta_prev
        I_delta_prev = I_delta
        I_delta = ((flux_linkage_sum - flux_linkage_sum_prev - 3*(R21*dt/2-L21)*I_delta_prev)/(3*(R21*dt/2+L21))) 
        #---- FIRST ESTIMATES ----
        
        #---- NEWTON ITERATIVE METHOD ----
        tolerance_value = 1*10**-5
        tolerance = np.array([[tolerance_value], 
                              [tolerance_value], 
                              [tolerance_value], 
                              [tolerance_value]])
        estimate = np.array([[Il_A], [Il_B], [Il_C], [I_delta]])
        solution = np.array([[999999], [999999], [999999], [999999]]) #choose any random improbable value
        while np.all(abs(solution - estimate) > tolerance):  
            function = np.array([[Il_A - 1/(1+dt/2*R/B_A)*(Il_A_prev + dt/2*((V_AN_prev - I_delta_prev*R - (I_delta_prev - I_delta_prev_prev)/dt*L)/B_A - Il_A_prev*R/B_A + (V_AN - I_delta*R - (I_delta - I_delta_prev)/dt*L)/B_A))], 
                                [Il_B - 1/(1+dt/2*R/B_B)*(Il_B_prev + dt/2*((V_BN_prev - I_delta_prev*R - (I_delta_prev - I_delta_prev_prev)/dt*L)/B_B - Il_B_prev*R/B_B + (V_BN - I_delta*R - (I_delta - I_delta_prev)/dt*L)/B_B))], 
                                [Il_C - 1/(1+dt/2*R/B_C)*(Il_C_prev + dt/2*((V_CN_prev - I_delta_prev*R - (I_delta_prev - I_delta_prev_prev)/dt*L)/B_C - Il_C_prev*R/B_C + (V_CN - I_delta*R - (I_delta - I_delta_prev)/dt*L)/B_C))], 
                                [I_delta - ((flux_linkage_sum - flux_linkage_sum_prev - 3*(R21*dt/2-L21)*I_delta_prev)/(3*(R21*dt/2+L21)))]])
            
            #Jacobian matrix
            J = np.array([[1, 0, 0, -1/(1+dt/2*R/B_A)*dt/2*(1/B_A*(-R-L/dt))], 
                          [0, 1, 0, -1/(1+dt/2*R/B_B)*dt/2*(1/B_B*(-R-L/dt))], 
                          [0, 0, 1, -1/(1+dt/2*R/B_C)*dt/2*(1/B_C*(-R-L/dt))], 
                          [-Lm_A_variable/(3*(R21*dt/2+L21)), -Lm_B_variable/(3*(R21*dt/2+L21)), -Lm_C_variable/(3*(R21*dt/2+L21)), 1]])
            
            if solution[0, 0] != 999999:
                estimate = solution.copy()
            solution = estimate - np.matmul(np.linalg.inv(J),function)
            I_delta = solution[3, 0]
            Il_A = solution[0, 0]
            Il_B = solution[1, 0]
            Il_C = solution[2, 0]
            I_A = Il_A + Lm_A_variable/Rm*(Il_A - Il_A_prev)/dt + I_delta 
            flux_linkage_A = flux_linkage_A_prev + Lm_A_variable*(Il_A - Il_A_prev)
            I_B = Il_B + Lm_B_variable/Rm*(Il_B - Il_B_prev)/dt + I_delta
            flux_linkage_B = flux_linkage_B_prev + Lm_B_variable*(Il_B - Il_B_prev)
            I_C = Il_C + Lm_C_variable/Rm*(Il_C - Il_C_prev)/dt + I_delta
            flux_linkage_C = flux_linkage_C_prev + Lm_C_variable*(Il_C - Il_C_prev)
            flux_linkage_sum = flux_linkage_A + flux_linkage_B + flux_linkage_C 
        #---- NEWTON ITERATIVE METHOD ----
        
        #Check sum of currents:
        I_0 = I_A + I_B + I_C
        
        #voltages
        Vm_A = Lm_A_variable*(Il_A - Il_A_prev)/dt
        Vm_B = Lm_B_variable*(Il_B - Il_B_prev)/dt
        Vm_C = Lm_C_variable*(Il_C - Il_C_prev)/dt
        # ΦA bus voltage
        V_AN_bus = V_AN - I_A*Rnetwork - (V_AN-I_A*R-Vm_A)/L*Lnetwork
        V_BN_bus = V_BN - I_B*Rnetwork - (V_BN-I_B*R-Vm_B)/L*Lnetwork
        # ΦB bus voltage
        V_BN_bus = V_BN - I_B*Rnetwork - (V_BN-I_B*R-Vm_B)/L*Lnetwork
        V_CN_bus = V_CN - I_C*Rnetwork - (V_CN-I_C*R-Vm_C)/L*Lnetwork
        # ΦC bus voltage
        V_CN_bus = V_CN - I_C*Rnetwork - (V_CN-I_C*R-Vm_C)/L*Lnetwork
        V_AN_bus = V_AN - I_A*Rnetwork - (V_AN-I_A*R-Vm_A)/L*Lnetwork
        
        #append lists
        V_AN_bus_list.append(V_AN_bus)
        V_AB_bus_rms_list.append(V_AB_bus_rms)
        V_AN_bus_rms_list.append(V_AN_bus_rms)
        V_BN_bus_list.append(V_BN_bus)
        V_BC_bus_rms_list.append(V_BC_bus_rms)
        V_BN_bus_rms_list.append(V_BN_bus_rms)
        V_CN_bus_list.append(V_CN_bus)
        V_CA_bus_rms_list.append(V_CA_bus_rms)
        V_CN_bus_rms_list.append(V_CN_bus_rms)
        flux_linkage_A_list.append(flux_linkage_A)
        flux_linkage_B_list.append(flux_linkage_B)
        flux_linkage_C_list.append(flux_linkage_C)
        I_A_list.append(I_A)
        I_B_list.append(I_B)
        I_C_list.append(I_C)
        V_AB_list.append(V_AB)
        V_BC_list.append(V_BC)
        V_CA_list.append(V_CA)
        V_AN_list.append(V_AN)
        V_BN_list.append(V_BN)
        V_CN_list.append(V_CN)
        I_delta_list.append(I_delta)
        I_0_list.append(I_0)
        I_A_rms_list.append(I_A_rms)
        I_B_rms_list.append(I_B_rms)
        I_C_rms_list.append(I_C_rms)
        t_list.append(t)
        
        #trapezoidal variables
        flux_linkage_sum_prev = flux_linkage_sum
        Il_A_prev =  Il_A
        V_AN_prev = V_AN
        flux_linkage_A_prev = flux_linkage_A
        Il_B_prev =  Il_B
        V_BN_prev = V_BN
        flux_linkage_B_prev = flux_linkage_B
        Il_C_prev =  Il_C
        V_CN_prev = V_CN
        flux_linkage_C_prev = flux_linkage_C
        
        t = t + dt #update time
    
    #create virtual arrays with less elements
    index_array = np.linspace(0, len(t_list)-1, int(len(t_list)/change_resolution))
    index_array = index_array.astype(int)
    t_virtual_list = [t_list[index] for index in index_array]
    V_AN_bus_virtual_list = [V_AN_bus_list[index] for index in index_array]
    V_BN_bus_virtual_list = [V_BN_bus_list[index] for index in index_array]
    V_CN_bus_virtual_list = [V_CN_bus_list[index] for index in index_array]
    V_AB_virtual_list = [V_AB_list[index] for index in index_array]
    V_AN_virtual_list = [V_AN_list[index] for index in index_array]
    V_BC_virtual_list = [V_BC_list[index] for index in index_array]
    V_BN_virtual_list = [V_BN_list[index] for index in index_array]
    V_CA_virtual_list = [V_CA_list[index] for index in index_array]
    V_CN_virtual_list = [V_CN_list[index] for index in index_array]
    V_AB_bus_rms_virtual_list = [V_AB_bus_rms_list[index] for index in index_array]
    V_AN_bus_rms_virtual_list = [V_AN_bus_rms_list[index] for index in index_array]
    V_BC_bus_rms_virtual_list = [V_BC_bus_rms_list[index] for index in index_array]
    V_BN_bus_rms_virtual_list = [V_BN_bus_rms_list[index] for index in index_array]
    V_CA_bus_rms_virtual_list = [V_CA_bus_rms_list[index] for index in index_array]
    V_CN_bus_rms_virtual_list = [V_CN_bus_rms_list[index] for index in index_array]
    I_A_virtual_list = [I_A_list[index] for index in index_array]
    I_B_virtual_list = [I_B_list[index] for index in index_array]
    I_C_virtual_list = [I_C_list[index] for index in index_array]
    I_A_rms_virtual_list = [I_A_rms_list[index] for index in index_array]
    I_B_rms_virtual_list = [I_B_rms_list[index] for index in index_array]
    I_C_rms_virtual_list = [I_C_rms_list[index] for index in index_array]
    dt_virtual = dt*change_resolution #frequency of sampling also changes
        
    #calculate frequency spectrum of inrush current signal: REDUCED
    sampling_freq = 1/dt
    number_samples = len(I_A_list[0:int(1/dt/dt*2)]) #2 cycles
    #ideal freq step
    freq_step_ideal_red = sampling_freq/number_samples #we want to reduce the freq_step. Therefore we have to reduce the sampling frequency
    #ideal freqs vector
    freqs_red = np.linspace(0, (number_samples-1)*freq_step_ideal_red, number_samples)
    freqs_scaling_red = freqs_red[0:int(number_samples/2+1)]
    #phase A
    I_A_list_fft_red = np.fft.fft(I_A_list[0:int(1/dt/dt*2)])
    I_A_list_fft_mag_red = np.abs(I_A_list_fft_red)/number_samples
    I_A_list_fft_mag_scaling_red = 2*I_A_list_fft_mag_red[0:int(number_samples/2+1)]
    I_A_list_fft_mag_scaling_red[0] = I_A_list_fft_mag_scaling_red[0]/2 #dc component
    #phase B
    I_B_list_fft_red = np.fft.fft(I_B_list[0:int(1/dt/dt*2)])
    I_B_list_fft_mag_red = np.abs(I_B_list_fft_red)/number_samples
    I_B_list_fft_mag_scaling_red = 2*I_B_list_fft_mag_red[0:int(number_samples/2+1)]
    I_B_list_fft_mag_scaling_red[0] = I_B_list_fft_mag_scaling_red[0]/2 #dc component
    #phase C
    I_C_list_fft_red = np.fft.fft(I_C_list[0:int(1/dt/dt*2)])
    I_C_list_fft_mag_red = np.abs(I_C_list_fft_red)/number_samples
    I_C_list_fft_mag_scaling_red = 2*I_C_list_fft_mag_red[0:int(number_samples/2+1)]
    I_C_list_fft_mag_scaling_red[0] = I_C_list_fft_mag_scaling_red[0]/2 #dc component
    
    n_cycles = int(t_stop/(1/f)) #number of cycles to represent
    #compute sliding rms voltage values, line AB and phase A
    e = int(1/f/dt_virtual) #initialize start position at the right-end of the window
    window_samples = int(1/f/dt_virtual) 
    while e < n_cycles*window_samples: #limit range of calculation
        index_array = np.linspace(e, e-window_samples+1, window_samples)
        index_array = index_array.astype(int)
        line_voltage_array = [V_AN_bus_virtual_list[index] - V_BN_bus_virtual_list[index] for index in index_array]
        phase_voltage_array = [V_AN_bus_virtual_list[index] for index in index_array]
        sum_squares = 0
        for pos in line_voltage_array:
            sum_squares = sum_squares + pos**2
        V_AB_bus_rms_virtual_list[e] = (sum_squares/(window_samples))**0.5
        sum_squares = 0
        for pos in phase_voltage_array:
            sum_squares = sum_squares + pos**2
        V_AN_bus_rms_virtual_list[e] = (sum_squares/(window_samples))**0.5
        e=e+1
    #complete with voltage before simulation, line AB and phase A
    e = 0
    #create array of one window from previous line and phase voltage
    index_array = index_array = np.linspace(0, window_samples-1, window_samples)
    index_array = index_array.astype(int)
    prev_line_voltage_array = [V_AB_virtual_list[index] for index in index_array]
    prev_phase_voltage_array = [V_AN_virtual_list[index] for index in index_array]
    #create array of one window for the first slice
    index_array = np.linspace(1, window_samples, window_samples)
    index_array = index_array.astype(int)
    line_voltage_array = [V_AN_bus_virtual_list[index] - V_BN_bus_virtual_list[index] for index in index_array]
    phase_voltage_array = [V_AN_bus_virtual_list[index] for index in index_array]
    #concatenate both arrays, line
    sum_array = np.append(prev_line_voltage_array,line_voltage_array)
    e = int(1/f/dt_virtual) 
    while e < 2*window_samples: #only two cycles
        index_array = np.linspace(e, e-window_samples+1, window_samples)
        index_array = index_array.astype(int)
        line_voltage_array = [sum_array[index] for index in index_array]
        sum_squares = 0
        for pos in line_voltage_array:
            sum_squares = sum_squares + pos**2
        V_AB_bus_rms_virtual_list[e-int(1/f/dt_virtual)] = (sum_squares/(window_samples))**0.5
        e=e+1
    #concatenate both arrays, phase
    sum_array = np.append(prev_phase_voltage_array,phase_voltage_array)
    e = int(1/f/dt_virtual) 
    while e < 2*window_samples: #only two cycles
        index_array = np.linspace(e, e-window_samples+1, window_samples)
        index_array = index_array.astype(int)
        phase_voltage_array = [sum_array[index] for index in index_array]
        sum_squares = 0
        for pos in phase_voltage_array:
            sum_squares = sum_squares + pos**2
        V_AN_bus_rms_virtual_list[e-int(1/f/dt_virtual)] = (sum_squares/(window_samples))**0.5
        e=e+1
    #compute sliding rms voltage values, line BC
    e = int(1/f/dt_virtual) #initialize start position at the right-end of the window
    window_samples = int(1/f/dt_virtual) 
    while e < n_cycles*window_samples: #limit range of calculation
        index_array = np.linspace(e, e-window_samples+1, window_samples)
        index_array = index_array.astype(int)
        line_voltage_array = [V_BN_bus_virtual_list[index] - V_CN_bus_virtual_list[index] for index in index_array]
        phase_voltage_array = [V_BN_bus_virtual_list[index] for index in index_array]
        sum_squares = 0
        for pos in line_voltage_array:
            sum_squares = sum_squares + pos**2
        V_BC_bus_rms_virtual_list[e] = (sum_squares/(window_samples))**0.5
        sum_squares = 0
        for pos in phase_voltage_array:
            sum_squares = sum_squares + pos**2
        V_BN_bus_rms_virtual_list[e] = (sum_squares/(window_samples))**0.5
        e=e+1
    #complete with voltage before simulation, line BC and phase B
    e = 0
    #create array of one window from previous line and phase voltage
    index_array = index_array = np.linspace(0, window_samples-1, window_samples)
    index_array = index_array.astype(int)
    prev_line_voltage_array = [V_BC_virtual_list[index] for index in index_array]
    prev_phase_voltage_array = [V_BN_virtual_list[index] for index in index_array]
    #create array of one window for the first slice
    index_array = np.linspace(1, window_samples, window_samples)
    index_array = index_array.astype(int)
    line_voltage_array = [V_BN_bus_virtual_list[index] - V_CN_bus_virtual_list[index] for index in index_array]
    phase_voltage_array = [V_BN_bus_virtual_list[index] for index in index_array]
    #concatenate both arrays, line
    sum_array = np.append(prev_line_voltage_array,line_voltage_array)
    e = int(1/f/dt_virtual) 
    while e < 2*window_samples: #only two cycles
        index_array = np.linspace(e, e-window_samples+1, window_samples)
        index_array = index_array.astype(int)
        line_voltage_array = [sum_array[index] for index in index_array]
        sum_squares = 0
        for pos in line_voltage_array:
            sum_squares = sum_squares + pos**2
        V_BC_bus_rms_virtual_list[e-int(1/f/dt_virtual)] = (sum_squares/(window_samples))**0.5
        e=e+1
    #concatenate both arrays, phase
    sum_array = np.append(prev_phase_voltage_array,phase_voltage_array)
    e = int(1/f/dt_virtual) 
    while e < 2*window_samples: #only two cycles
        index_array = np.linspace(e, e-window_samples+1, window_samples)
        index_array = index_array.astype(int)
        phase_voltage_array = [sum_array[index] for index in index_array]
        sum_squares = 0
        for pos in phase_voltage_array:
            sum_squares = sum_squares + pos**2
        V_BN_bus_rms_virtual_list[e-int(1/f/dt_virtual)] = (sum_squares/(window_samples))**0.5
        e=e+1
    #compute sliding rms voltage values, line CA and phase C
    e = int(1/f/dt_virtual) #initialize start position at the right-end of the window
    window_samples = int(1/f/dt_virtual) 
    while e < n_cycles*window_samples: #limit range of calculation
        index_array = np.linspace(e, e-window_samples+1, window_samples)
        index_array = index_array.astype(int)
        line_voltage_array = [V_CN_bus_virtual_list[index] - V_AN_bus_virtual_list[index] for index in index_array]
        phase_voltage_array = [V_CN_bus_virtual_list[index] for index in index_array]
        sum_squares = 0
        for pos in line_voltage_array:
            sum_squares = sum_squares + pos**2
        V_CA_bus_rms_virtual_list[e] = (sum_squares/(window_samples))**0.5
        sum_squares = 0
        for pos in phase_voltage_array:
            sum_squares = sum_squares + pos**2
        V_CN_bus_rms_virtual_list[e] = (sum_squares/(window_samples))**0.5
        e=e+1
    #complete with voltage before simulation, line CA
    e = 0
    #create array of one window from previous line and phase voltage
    index_array = index_array = np.linspace(0, window_samples-1, window_samples)
    index_array = index_array.astype(int)
    prev_line_voltage_array = [V_CA_virtual_list[index] for index in index_array]
    prev_phase_voltage_array = [V_CN_virtual_list[index] for index in index_array]
    #create array of one window for the first slice
    index_array = np.linspace(1, window_samples, window_samples)
    index_array = index_array.astype(int)
    line_voltage_array = [V_CN_bus_virtual_list[index] - V_AN_bus_virtual_list[index] for index in index_array]
    phase_voltage_array = [V_CN_bus_virtual_list[index] for index in index_array]
    #concatenate both arrays, line
    sum_array = np.append(prev_line_voltage_array,line_voltage_array)
    e = int(1/f/dt_virtual) 
    while e < 2*window_samples: #only two cycles
        index_array = np.linspace(e, e-window_samples+1, window_samples)
        index_array = index_array.astype(int)
        line_voltage_array = [sum_array[index] for index in index_array]
        sum_squares = 0
        for pos in line_voltage_array:
            sum_squares = sum_squares + pos**2
        V_CA_bus_rms_virtual_list[e-int(1/f/dt_virtual)] = (sum_squares/(window_samples))**0.5
        e=e+1
    #concatenate both arrays, phase
    sum_array = np.append(prev_phase_voltage_array,phase_voltage_array)
    e = int(1/f/dt_virtual) 
    while e < 2*window_samples: #only two cycles
        index_array = np.linspace(e, e-window_samples+1, window_samples)
        index_array = index_array.astype(int)
        phase_voltage_array = [sum_array[index] for index in index_array]
        sum_squares = 0
        for pos in phase_voltage_array:
            sum_squares = sum_squares + pos**2
        V_CN_bus_rms_virtual_list[e-int(1/f/dt_virtual)] = (sum_squares/(window_samples))**0.5
        e=e+1
        
    #### CURRENT RMS CALCULATION ####
    n_cycles = int(t_stop/(1/f)) #number of cycles to represent
    #compute sliding rms current values, phase A
    e = int(1/f/dt_virtual) #initialize start position at the right-end of the window
    window_samples = int(1/f/dt_virtual) 
    while e < n_cycles*window_samples: #limit range of calculation
        index_array = np.linspace(e, e-window_samples+1, window_samples)
        index_array = index_array.astype(int)
        phase_current_array = [I_A_virtual_list[index] for index in index_array]
        sum_squares = 0
        for pos in phase_current_array:
            sum_squares = sum_squares + pos**2
        I_A_rms_virtual_list[e] = (sum_squares/(window_samples))**0.5
        e=e+1
    #complete with voltage before simulation, phase A
    e = 0
    #create array of one window from previous phase current
    index_array = index_array = np.linspace(0, window_samples-1, window_samples)
    index_array = index_array.astype(int)
    prev_phase_current_array = np.zeros(len(index_array))
    #create array of one window for the first slice
    index_array = np.linspace(1, window_samples, window_samples)
    index_array = index_array.astype(int)
    phase_current_array = [I_A_virtual_list[index] for index in index_array]
    #concatenate both arrays, phase
    sum_array = np.append(prev_phase_current_array,phase_current_array)
    e = int(1/f/dt_virtual) 
    while e < 2*window_samples: #only two cycles
        index_array = np.linspace(e, e-window_samples+1, window_samples)
        index_array = index_array.astype(int)
        phase_current_array = [sum_array[index] for index in index_array]
        sum_squares = 0
        for pos in phase_current_array:
            sum_squares = sum_squares + pos**2
        I_A_rms_virtual_list[e-int(1/f/dt_virtual)] = (sum_squares/(window_samples))**0.5
        e=e+1
    #compute sliding rms current values, phase B
    e = int(1/f/dt_virtual) #initialize start position at the right-end of the window
    window_samples = int(1/f/dt_virtual) 
    while e < n_cycles*window_samples: #limit range of calculation
        index_array = np.linspace(e, e-window_samples+1, window_samples)
        index_array = index_array.astype(int)
        phase_current_array = [I_B_virtual_list[index] for index in index_array]
        sum_squares = 0
        for pos in phase_current_array:
            sum_squares = sum_squares + pos**2
        I_B_rms_virtual_list[e] = (sum_squares/(window_samples))**0.5
        e=e+1
    #complete with voltage before simulation, phase B
    e = 0
    #create array of one window from previous phase current
    index_array = index_array = np.linspace(0, window_samples-1, window_samples)
    index_array = index_array.astype(int)
    prev_phase_current_array = np.zeros(len(index_array))
    #create array of one window for the first slice
    index_array = np.linspace(1, window_samples, window_samples)
    index_array = index_array.astype(int)
    phase_current_array = [I_B_virtual_list[index] for index in index_array]
    #concatenate both arrays, phase
    sum_array = np.append(prev_phase_current_array,phase_current_array)
    e = int(1/f/dt_virtual) 
    while e < 2*window_samples: #only two cycles
        index_array = np.linspace(e, e-window_samples+1, window_samples)
        index_array = index_array.astype(int)
        phase_current_array = [sum_array[index] for index in index_array]
        sum_squares = 0
        for pos in phase_current_array:
            sum_squares = sum_squares + pos**2
        I_B_rms_virtual_list[e-int(1/f/dt_virtual)] = (sum_squares/(window_samples))**0.5
        e=e+1
    #compute sliding rms current values, phase C
    e = int(1/f/dt_virtual) #initialize start position at the right-end of the window
    window_samples = int(1/f/dt_virtual) 
    while e < n_cycles*window_samples: #limit range of calculation
        index_array = np.linspace(e, e-window_samples+1, window_samples)
        index_array = index_array.astype(int)
        phase_current_array = [I_C_virtual_list[index] for index in index_array]
        sum_squares = 0
        for pos in phase_current_array:
            sum_squares = sum_squares + pos**2
        I_C_rms_virtual_list[e] = (sum_squares/(window_samples))**0.5
        e=e+1
    #complete with voltage before simulation, phase A
    e = 0
    #create array of one window from previous phase current
    index_array = index_array = np.linspace(0, window_samples-1, window_samples)
    index_array = index_array.astype(int)
    prev_phase_current_array = np.zeros(len(index_array))
    #create array of one window for the first slice
    index_array = np.linspace(1, window_samples, window_samples)
    index_array = index_array.astype(int)
    phase_current_array = [I_C_virtual_list[index] for index in index_array]
    #concatenate both arrays, phase
    sum_array = np.append(prev_phase_current_array,phase_current_array)
    e = int(1/f/dt_virtual) 
    while e < 2*window_samples: #only two cycles
        index_array = np.linspace(e, e-window_samples+1, window_samples)
        index_array = index_array.astype(int)
        phase_current_array = [sum_array[index] for index in index_array]
        sum_squares = 0
        for pos in phase_current_array:
            sum_squares = sum_squares + pos**2
        I_C_rms_virtual_list[e-int(1/f/dt_virtual)] = (sum_squares/(window_samples))**0.5
        e=e+1
        
    # REPRESENTS HARMONIC SPECTRUM --> REDUCED
    ax8.plot(freqs_scaling_red, I_A_list_fft_mag_scaling_red, freqs_scaling_red, I_B_list_fft_mag_scaling_red, '--')
    ax8.legend(["I_A freq domain", "I_B freq domain"], loc='right')
    ax8.set_xlabel("Frequency [Hz]")
    ax8.set_ylabel("Magitude [A]")
    ax8.set_xlim([0, 600])
    ax8.set_title("Harmonic content", pad=15)
    ax8.grid()
    try:
        ax8.figure.savefig(r'results/Harmonic_spectrum.png')
    except:
        print("Error saving image file")
    
    # REPRESENTS GRAPH OF INRUSH CURRENT 
    ax1.plot(t_list, I_A_list, t_list, I_B_list, t_list, I_C_list)
    ax1.legend(["I_A", "I_B", "I_C"])
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Current [A]")
    ax1.set_title("Inrush current", pad=15)
    ax1.grid()
    try:
        ax1.figure.savefig(r'results/Inrush_current.png')
    except:
        print("Error saving image file")
    
    # REPRESENTS GRAPH OF FLUX-LINKAGE ACROSS THE MAGNETIZING INDUCTANCE
    ax2.plot(t_list, flux_linkage_A_list, t_list, flux_linkage_B_list, t_list, flux_linkage_C_list)
    ax2.legend(["λ_A", "λ_B", "λ_C"])
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Flux-linkage [Wb]")
    ax2.set_title("Flux-linkage across the magnetizing inductance", pad=15)
    
    # REPRESENTS GRAPH OF BUS LINE RMS VOLTAGE
    ax3.plot(t_virtual_list, V_AB_bus_rms_virtual_list, t_virtual_list, V_BC_bus_rms_virtual_list, t_virtual_list, V_CA_bus_rms_virtual_list)
    ax3.legend(["V_AB_bus (rms)", "V_BC_bus (rms)", "V_CA_bus (rms)"])
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Voltage [V]")
    ax3.set_xlim([0, 1/f*n_cycles]) #considered less time that total simulation for more efficiency
    ax3.set_title("RMS line voltage drop at energizing bus", pad=15)
    ax3.grid()
    try:
        ax3.figure.savefig(r'results/line_RMS_bus_voltage_curve.png')
    except:
        print("Error saving image file")
    
    # REPRESENTS PHASE VOLTAGES
    ax4.plot(t_list, V_AN_list, t_list, V_BN_list, t_list, V_CN_list)
    ax4.legend(["V_AN", "V_BN", "V_CN"])
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Voltage [V]")
    ax4.set_title("Phase voltages of source", pad=15)
    
    # REPRESENTS GRAPH OF BUS PHASE VOLTAGE
    ax5.plot(t_list, V_AN_bus_list, t_list, V_BN_bus_list, t_list, V_CN_bus_list)
    ax5.legend(["V_AN_bus", "V_BN_bus", "V_CN_bus"])
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Voltage [V]")
    ax5.set_title("Phase voltages of energizing bus", pad=15)
    
    # REPRESENTS I DELTA
    ax6.plot(t_list, I_delta_list)
    ax6.legend(["I_delta"])
    ax6.set_xlabel("Time [s]")
    ax6.set_ylabel("Current [A]")
    ax6.set_title("Delta current in LV winding, seen from the HV winding", pad=15)
    
    # REPRESENTS I_0
    ax7.plot(t_list, I_0_list)
    ax7.legend(["I_0"])
    ax7.set_xlabel("Time [s]")
    ax7.set_ylabel("Current [A]")
    ax7.set_title("Sum of phase currents in HV winding", pad=15)
    
    # PLOTS SATURATION CURVES
    printSaturationCurve(V_F, Lm, L_A, f, K, N)
    
    return [t_virtual_list, I_A_virtual_list, I_B_virtual_list, I_C_virtual_list, V_AB_bus_rms_virtual_list, V_BC_bus_rms_virtual_list, V_CA_bus_rms_virtual_list, freqs_scaling_red, I_A_list_fft_mag_scaling_red, I_B_list_fft_mag_scaling_red, V_AN_bus_rms_virtual_list, V_BN_bus_rms_virtual_list, V_CN_bus_rms_virtual_list, I_A_rms_virtual_list, I_B_rms_virtual_list, I_C_rms_virtual_list]

if __name__ == "__main__":
    
    # INPUTS --> USED DATA IN RESEARCH PAPER --> In theory: ASK USER, GRAB FROM PSSE, OR ASSUME VALUES

    V_L = 11000 #V #tension de linea rms
    V2_L = 690 #V #tension de linea winding 2 rms
    L1 = 0.00244 #H 
    L2 = 9.6335*10**-6 #H
    Lm = 69.9823 #H
    L_A = 0.07336 #H #Air core inductance. Assume 2*L (or 10-20 smaller than Lm)
    R1 = 0.0864 #ohm
    R2 = 0.00034 #ohm
    Rm = 39465 #ohm
    phi_0 = -90*2*math.pi/360 #rad
    f = 50 #Hz
    w = 2*math.pi*f #rad/s
    flux_linkage_A_0 = 0 #Wb
    flux_linkage_B_0 = 0 #Wb
    flux_linkage_C_0 = 0 #Wb
    Rnetwork = 0 #ohm
    Lnetwork = 0.022 #H 
    K = 1.15 #pu #Knee point
    N = 10 #piecewise linear sections in saturation curve
    dt = 5*10**-6 #s #input
    t_stop = 0.2 #s #input
    change_resolution = 8 #divides by that number the elemets in the original list to create virtual arrays
    bus_energ_voltage_pu = 1
    V_source = 11000
    
    [t_virtual_list, I_A_virtual_list, I_B_virtual_list, I_C_virtual_list, V_AB_bus_rms_virtual_list, V_BC_bus_rms_virtual_list, V_CA_bus_rms_virtual_list, freqs_scaling_red, I_A_list_fft_mag_scaling_red, I_B_list_fft_mag_scaling_red, V_AN_bus_rms_virtual_list, V_BN_bus_rms_virtual_list, V_CN_bus_rms_virtual_list, I_A_rms_virtual_list, I_B_rms_virtual_list, I_C_rms_virtual_list] = EnergizationWyegDelta(V_L, V2_L, L1, L2, Lm, L_A, R1, R2, Rm, phi_0, f, flux_linkage_A_0, flux_linkage_B_0, flux_linkage_C_0, Rnetwork, Lnetwork, K, N, dt, t_stop, change_resolution, bus_energ_voltage_pu, V_source)
    plt.show()