import sys
import os

cdir = os.getcwd() #get current directory
os.chdir(cdir) #make sure you are in the current directory
sys.path.append(cdir) #append directory to enable import of functions from current directory files

import numpy as np

from params import setParams

def run(params=None, fp=np.array([0.0, 0.01]), wavenumber=1, itype='rungekutta'):
    """
            
    This is the model implementation for the continuum Wilson-Cowan model, called mtype = 'activity'
    or else the continuum Amari model, called mtype = 'voltage'.
    System type: Integro-Differential equations with temporal and spatial component => dim(system)=2d
    IDEs for excitatory & inhibitory populations (coupled)
    Spatial dimension: dim(x)=1d, arranged on a ring 
    -> all-to-all same connectivity, determined by spread, omitting boundary conditions
            
        
    List of possible keys+values in params-dict:

    parameters:
    :w_ee: excitatory to excitatory coupling, float
    :w_ei: inhibitory to excitatory coupling, float
    :w_ie: excitatory to inhibitory coupling, float
    :w_ii: inhibitory to inhibitory coupling, float
        
    :tau_e: excitatory membrane time constant, float
    :tau_i: inhibitory membrane time constant, float
        
    :beta_e: excitatory gain (in sigmoidal transfer function), float
    :beta_i: inhibitory gain (in sigmoidal transfer function), float
    :mu_e: excitatory threshold (in sigmoidal transfer function), float
    :mu_i: inhibitory threshold (in sigmoidal transfer function), float
        
    :I_e: external input current to excitatory population, float
    :I_i: external input current to inhibitory population, float
        
    :kernel: what function used to determine spatial kernel, string, options are 'gaussian' or 'exponential'
    :sigma_e: characterizes the spatial extent of the excitatory to [...] connectivity, float
    :sigma_i: characterizes the spatial extent of the inhibitory to [...] connectivity, float

    mechanism parameters:
    :beta_a: mechanism gain in sigmoidal transfer function, float (>0 if adaptation, <0 if h-current)
    :mu_a: mechanism threshold in sigmoidal transfer function, float
    :tau_i: mechanism time constant, float
    :b: mechanism strength, float  (>0 if adaptation, <0 if h-current)
        
    temporal components:
    :dt: integration time step, float -> observe ms, therefore, setting e.g. dt=0.1, means we look at every 10th ms.
    :start_t: start of time intervall, integer
    :end_t: end of time intervall, integer
        
    spatial components:
    :n: number of pixels/positions on ring, integer
    :length: length of total circumference of ring, float 
             (remark: max. distance from pixel to furthest away can bi maximally length/2)
    :c: velocity of activity in [m/s], float -> transformed into mm/s in py.params.setParams()
        
    created by given params:
    :x: array of distances from one pixel to all other pixels (same distances to left and right, omit boundary effects), array
    :dx: spatial integration step, determined by length and n, float
    :ke: kernel values (i.e. connectivity strengths) from excitatory population of a pixel to all other pixels, 
         determined by x, array
    :ki: kernel values (i.e. connectivity strengths) from excitatory population of a pixel to all other pixels, 
         determined by x, array
    :ke_fft: Fast Fourier Transform of ke by np.fft.fft, array
    :ki_fft: Fast Fourier Transform of ki by np.fft.fft, array 
    :time: array of time intervall, array
    :delay: temporal delay from one pixel to another, determined by x,c and dt, array    
    """

    params = setParams(params)
    
    return runIntegration(params, fp=fp, wavenumber=wavenumber, itype=itype)

def kernSeed(array, kernel, seed_amp):
    array += kernel*seed_amp
    return array

#pseud-random-number-generator for reproducible initialisations
def prngSeed(shape, fp, std, seed=42):

    # Set the seed for reproducibility
    np.random.seed(seed)
    
    # Generate random numbers with a normal distribution around 0
    noise = np.random.normal(loc=0, scale=std, size=shape)
    # Add the fixed value to the noise
    jittered_array = fp + noise

    return jittered_array 

def sinusSeed(array, length, wavenumber, seed_amp, x):
    wavelength = 2*np.pi / wavenumber
    m = length/wavelength
    wave_in_pi = (2*np.pi*m) / length
    x = np.roll(np.fft.fftshift(x), -1)
    phi = np.pi
    
    array += seed_amp * (np.cos(wave_in_pi * x) + 0.2 * np.sin(2 * wave_in_pi * x + phi))
    
    return array

def runIntegration(params, fp=np.array([0.0, 0.01]), wavenumber=1, itype='rungekutta'):
    
    """
    Before we can run the integration-loop, we have to set the parameters and call the integration by them 
    s.t. nothing's accidentally overwritten.
    """
    
    #membrane time constants [no units yet]
    tau_e = params.tau_e
    tau_i = params.tau_i
    
    #coupling weights (determining the dominant type of cells, e.g. locally excitatory...)
    w_ee = params.w_ee
    w_ei = params.w_ei
    w_ie = params.w_ie
    w_ii = params.w_ii
    
    #threshold and gain factors for the sigmoidal activation functions 
    beta_e = params.beta_e
    beta_i = params.beta_i
    mu_e = params.mu_e
    mu_i = params.mu_i
    
    # # - - mechanism parameters - - # #
    
    #transfer function
    beta_a = params.beta_a
    mu_a = params.mu_a
    
    #strength and time constant
    b = params.b
    tau_a = params.tau_a
    
    # # - - - - # #
    
    #external input currents: oscillatory state for default params
    I_e = params.I_e
    I_i = params.I_i

    #temporal 
    dt = params.dt
    
    #spatial
    n = params.n
    length = params.length
    c = params.c
    
    x = params.x
    dx = params.dx
    time = params.time
    
    ke = params.ke
    ki = params.ki
    
    ke_fft = params.ke_fft
    ki_fft = params.ki_fft
    
    delayed = params.delayed
    delay = params.delay
    
    seed =  params.seed
    seed_amp =  params.seed_amp
    seed_func = params.seed_func
    
    comparison = fp==[0.0,0.01]
    
    if all(comparison):
        init_exc = fp
        init_inh = fp
        init_adaps = fp
    else:
        a_fp = 1/(1+np.exp(-beta_a*(fp[0]-mu_a)))
        init_exc = [fp[0]-1e-10, fp[0]+1e-10]
        init_inh = [fp[1]-1e-10, fp[1]+1e-10]
        init_adaps = [a_fp-1e-10, a_fp+1e-10]
        
    if seed and not all(comparison): 
        ue_init = np.ones((len(time),n))*fp[0] #leads to [rows, columns] = [time, pixels (space)]
        ui_init = np.ones((len(time),n))*fp[1]
        adaps_init = np.ones((len(time),n))*a_fp

        #usually now I distinguish different seed functions but I only have one yet, so ...
        if seed_func == 'kern':
            ue_init[0] = kernSeed(ue_init[0], ke, seed_amp)
            ui_init[0] = kernSeed(ui_init[0], ki, seed_amp)
            adaps_init[0] = kernSeed(adaps_init[0], ke, seed_amp)
        elif seed_func == 'sinus':
            ue_init[0] = sinusSeed(ue_init[0], length, wavenumber, seed_amp, x)
            ui_init[0] = sinusSeed(ui_init[0], length, wavenumber, seed_amp, x)
            adaps_init[0] = sinusSeed(adaps_init[0], length, wavenumber, seed_amp, x)
        else:
            ue_init[0] = prngSeed(n, fp[0], seed_amp, 42)
            ui_init[0] = prngSeed(n, fp[1], seed_amp, 42)
            adaps_init[0] = prngSeed(n, a_fp, seed_amp, 42)
    else:
        #the initialisation I have to make to start the integration
        ue_init = np.zeros((len(time),n)) #leads to [rows, columns] = [time, pixels (space)]
        ui_init = np.zeros((len(time),n))
        adaps_init = np.zeros((len(time),n))
        ue_init[0]=np.random.uniform(init_exc[0], init_exc[1], n)
        ui_init[0]=np.random.uniform(init_inh[0], init_inh[1], n)
        adaps_init[0]=np.random.uniform(init_adaps[0], init_adaps[1], n)
    
    
    integrate = globals()['integrate_runge_kutta']
    
    ue, ui, adaps =  integrate(tau_e, tau_i,
                        w_ee, w_ei, w_ie, w_ii,
                        beta_e, beta_i, mu_e, mu_i,
                        I_e, I_i,
                        beta_a, mu_a, b, tau_a, adaps_init,
                        dt, time, 
                        n, length, c, x, dx, 
                        ke, ki, ke_fft, ki_fft,
                        ue_init, ui_init)
    
    
    return ue, ui, adaps

def integrate_runge_kutta(tau_e, tau_i,
                      w_ee, w_ei, w_ie, w_ii,
                      beta_e, beta_i, mu_e, mu_i,
                      I_e, I_i,
                      beta_a, mu_a, b, tau_a, adaps,
                      dt, time, 
                      n, length, c, x, dx,
                      ke, ki, ke_fft, ki_fft,
                      ue, ui):
    
    """"
    NOTE: The Runge-Kutta Implementation - as of right now - does NOT work with delays!!!
    """

    def Fe(x):
        return 1 / (1 + np.exp(-beta_e * (x-mu_e)))

    def Fi(x):
        return 1 / (1 + np.exp(-beta_i * (x-mu_i)))

    def Fa(x):
        return 1 / (1 + np.exp(-beta_a * (x-mu_a)))
    
    def compute_rhs(ee, conv_e, ii, conv_i, ah):
        rhs_adaps = (1 / tau_a) * (-ah + Fa(ee))
        rhs_e = (1 / tau_e) * (-ee + Fe(w_ee * conv_e - w_ei * conv_i - b * ah + I_e))
        rhs_i = (1 / tau_i) * (-ii + Fi(w_ie * conv_e - w_ii * conv_i + I_i))
        return rhs_e, rhs_i, rhs_adaps


    N = len(ke)
    
    ke_padded = np.pad(ke, (0, N-1), mode='constant')
    ki_padded = np.pad(ki, (0, N-1), mode='constant')

    ke_fft = np.fft.fft(ke_padded)
    ki_fft = np.fft.fft(ki_padded)

    for t in range(1, int(len(time))):
        value = 0.1
        #enforce Dirichlet boundary conditions
        ue[t-1][0] = value
        ue[t-1][-1] = value

        ui[t-1][0] = value
        ui[t-1][-1] = value

        adaps[t-1][0] = value
        adaps[t-1][-1] = value

        N = len(ue[t-1])
    
        #Zero-pad both the signal and kernel to 2N-1
        ue_padded = np.pad(ue[t-1], (0, N-1), mode='constant')
        ui_padded = np.pad(ui[t-1], (0, N-1), mode='constant')

        ve = np.fft.fft(ue_padded)
        vi = np.fft.fft(ui_padded)

        Le = ke_fft * ve
        Li = ki_fft * vi

        conv_e = np.fft.ifft(Le).real
        conv_i = np.fft.ifft(Li).real
        conv_e = np.real(conv_e[:N])
        conv_i = np.real(conv_i[:N])

        #collect 4 order terms of rhs for Runge-Kutta
        k1_e, k1_i, k1_adaps = compute_rhs(ue[t-1], conv_e, ui[t-1], conv_i, adaps[t-1])
        k2_e, k2_i, k2_adaps = compute_rhs(ue[t-1] + 0.5 * dt * k1_e, conv_e, ui[t-1] + 0.5 * dt * k1_i, conv_i, adaps[t-1] + 0.5 * dt * k1_adaps)
        k3_e, k3_i, k3_adaps = compute_rhs(ue[t-1] + 0.5 * dt * k2_e, conv_e, ui[t-1] + 0.5 * dt * k2_i, conv_i, adaps[t-1] + 0.5 * dt * k2_adaps)
        k4_e, k4_i, k4_adaps = compute_rhs(ue[t-1] + dt * k3_e, conv_e, ui[t-1] + dt * k3_i, conv_i, adaps[t-1] + dt * k3_adaps)

        ue[t] = ue[t-1] + (dt / 6) * (k1_e + 2 * k2_e + 2 * k3_e + k4_e)
        ui[t] = ui[t-1] + (dt / 6) * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)
        adaps[t] = adaps[t-1] + (dt / 6) * (k1_adaps + 2 * k2_adaps + 2 * k3_adaps + k4_adaps)
        if t==1 and any(adaps[t] <= 0):
            adaps[t] = np.where(adaps[t] >= 0, adaps[t], 0)


    return ue, ui, adaps