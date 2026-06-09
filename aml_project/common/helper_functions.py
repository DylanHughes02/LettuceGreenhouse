import numpy as np
import statistics

from scipy import signal
from scipy.io import loadmat

"""
    Helper functions for the greenhouse climate lettuce growth model.
"""

def normalise(x, minx, maxx):
    return 2*(x-minx)/(maxx-minx) - 1

def denormalise(normx, minx, maxx):
    return (normx+1)*(maxx-minx)/2 + minx

def normalise_array(obs, minobs, maxobs):
    norm_obs = np.zeros(len(obs))
    for i, a in enumerate(obs): norm_obs[i] += normalise(a, minobs[i], maxobs[i])
    return norm_obs

def denormalise_array(action: np.ndarray, minas, maxas)-> np.ndarray[np.float64]:
    '''
    denormalises the action array to the original action space
    
    Parameters
    ----------
    action : array
        action array to be denormalised
    minas : array
        minimum action values
    maxas : array
        maximum action values
        
    Returns
    -------
    denorm_action : array
    '''
    denorm_action = np.zeros(len(action))
    for i, a in enumerate(action): denorm_action[i] += denormalise(a, minas[i], maxas[i])
    return denorm_action

def denormalize_action(action, min_action, max_action):
    return (action+1) * (max_action - min_action)/2 + min_action

def weather2ppmrh(d):
    """
    Converts raw weather data of CO2 concentration and vapour pressure deficit.
    to CO2 concentration in ppm and relative humidity in %.
    """
    # convert CO2 concentration to pm
    d[:,1] = co2dens2ppm(d[:, 2], d[:, 1])*1e-3
    # convert vapour pressure deficit to relative humidity
    d[:,3] = vaporDens2rh(d[:, 2], d[:, 3])
    return d

def rh2vaporDens(temp,rh):
    
    # constants
    R = 8.3144598 # molar gas constant [J mol^{-1} K^{-1}]
    C2K = 273.15 # conversion from Celsius to Kelvin [K]
    Mw = 18.01528e-3 # molar mass of water [kg mol^-{1}]

    # parameters used in the conversion
    c = np.array([610.78, 238.3, 17.2694, -6140.4, 273, 28.916])

    satP = c[0]*np.exp(c[2]*np.divide(temp,(temp+c[1])))
    # Saturation vapor pressure of air in given temperature [Pa]
    
    pascals=(rh/100)*satP; # Partial pressure of vapor in air [Pa]
    
    # convert to density using the ideal gas law pV=nRT => n=pV/RT 
    # so n=p/RT is the number of moles in a m^3, and Mw*n=Mw*p/(R*T) is the 
    # number of kg in a m^3, where Mw is the molar mass of water.
    
    vaporDens = np.divide(pascals*Mw,(R*(temp+C2K)))
        
    return vaporDens

def vaporDens2rh(temp, vaporDens):
    """
    vaporDens2rh Convert vapor density [kg{H2O} m^{-3}] to relative humidity [%]

    Usage:
    rh = vaporDens2rh(temp, vaporDens)
    Inputs:
    temp        given temperatures [°C] (numeric vector)
    vaporDens   absolute humidity [kg{H20} m^{-3}] (numeric vector)
    Inputs should have identical dimensions
    Outputs:
    rh          relative humidity [%] between 0 and 100 (numeric vector)

    Calculation based on 
    http://www.conservationphysics.org/atmcalc/atmoclc2.pdf

    David Katzin, Wageningen University
    david.katzin@wur.nl
    """
    # constants
    # molar gas constant [J mol^{-1} K^{-1}]
    R = 8.3144598 
    # conversion from Celsius to Kelvin [K]
    C2K = 273.15  
    # molar mass of water [kg mol^-{1}]
    Mw = 18.01528e-3  
    
    # parameters used in the conversion
    # default value is [610.78 238.3 17.2694 -6140.4 273 28.916]
    p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]
    
    # Saturation vapor pressure of air in given temperature [Pa]
    satP = p[0]*np.exp(p[2]*temp/(temp+p[1])) 
    # convert to relative humidity using the ideal gas law pV=nRT => n=pV/RT 
    # so n=p/RT is the number of moles in a m^3, and Mw*n=Mw*p/(R*T) is the 
    # number of kg in a m^3, where Mw is the molar mass of water.
    relhumid = 100*R*(temp+C2K)/(Mw*satP)*vaporDens
    # if np.isinf(relhumid).any():
    #     print(temp, vaporDens)
    return np.clip(relhumid, a_min=0, a_max=100)

def co2dens2ppm(temp, dens):
    """
    CO2DENS2PPM Convert CO2 density [kg m^{-3}] to molar concetration [ppm] 

    Usage: 
    ppm = co2dens2ppm(temp, dens)
    Inputs:
    temp        given temperatures [°C] (numeric vector)
    dens        CO2 density in air [kg m^{-3}] (numeric vector)
    Inputs should have identical dimensions
    Outputs:
    ppm         Molar concentration of CO2 in air [ppm] (numerical vector)

    calculation based on ideal gas law pV=nRT, pressure is assumed to be 1 atm

    David Katzin, Wageningen University
    david.katzin@wur.nl
    """

    # molar gas constant [J mol^{-1} K^{-1}]
    R = 8.3144598
    # conversion from Celsius to Kelvin [K]
    C2K = 273.15
    # molar mass of CO2 [kg mol^-{1}]
    M_CO2 = 44.01e-3
    # pressure (assumed to be 1 atm) [Pa]
    P = 101325

    return 1e6*R*(temp+C2K)*dens/(P*M_CO2)

def co2ppm2dens(temp, ppm):
    """
    CO2PPM2DENS Convert CO2 molar concetration [ppm] to density [kg m^{-3}]
    Usage:
      co2Dens = co2ppm2dens(temp, ppm) 
    Inputs:
      temp        given temperatures [°C] (numeric vector)
      ppm         CO2 concetration in air (ppm) (numeric vector)
      Inputs should have identical dimensions
    Outputs:
      co2Dens     CO2 concentration in air [kg m^{-3}] (numeric vector)

    Calculation based on ideal gas law pV=nRT, with pressure at 1 atm

    David Katzin, Wageningen University
    david.katzin@wur.nl
    """
    R = 8.3144598 # molar gas constant [J mol^{-1} K^{-1}]
    C2K = 273.15 # conversion from Celsius to Kelvin [K]
    M_CO2 = 44.01e-3 # molar mass of CO2 [kg mol^-{1}]
    P = 101325 # pressure (assumed to be 1 atm) [Pa]
    
    # number of moles n=m/M_CO2 where m is the mass [kg] and M_CO2 is the
    # molar mass [kg mol^{-1}]. So m=p*V*M_CO2*P/RT where V is 10^-6*ppm    
    return  P*1e-6*ppm*M_CO2/(R*(temp+C2K))

def load_disturbances(c, L, h, nd, Np, startDay, weather_data_dir):
    """
    #### Description:
    Loads weather disturbances for a greenhouse simulation.

    #### Inputs:
    1. `c` (float): Constant representing a specific time unit, used in time conversions.
   
    2. `L` (float): Duration of the simulation in seconds.

    3. `h` (float): Time step or sampling period in seconds for the simulation.

    4. `nd` (int): Number of disturbances or weather-related parameters.

    5. `Np` (int): Number of time periods per disturbance.

    6. `startDay` (float): Start day for the simulation.

    7. `weather_data_dir` (str): Directory path or location of the weather data file to be loaded.

    #### Returns:
    - `d` (ndarray): Transposed array containing resampled and adjusted weather disturbance data for the greenhouse simulation.

    #### Processing Steps:
    1. **Data Loading:** Loads weather data from the specified directory using `loadmat`.
    2. **Time Processing:** Processes time-related information from the loaded dataset.
    3. **Parameter Extraction:** Extracts weather parameters like global radiation, temperature, CO2 concentration, and humidity.
    4. **Resampling:** Adjusts and resamples the data to fit the simulation requirements.
    5. **Data Adjustment:** Ensures non-negative values for specific parameters (e.g., radiation).
    6. **Returns:** Provides the processed and adjusted weather disturbance data suitable for simulation.

    """
    c       = 86400 # amount of seconds in a day
    nDays   = L/c      # number of days in the simulation L i
    #pdb.set_trace()
    D       = loadmat(weather_data_dir)
    D       = D['d']
    t       = D[:,0]                            # Time [days]
    t       = t - t[0]

    dt      = np.mean(np.diff(t))       # Sample period data [days]
    Ns      = int(np.ceil(nDays/dt))    # Number of samples we need
    N0      = int(np.ceil(startDay/dt)) # Start index

    if Ns > len(t):
       print(' ')
       print('Not enough samples in the data.')
       print(' ')

    if Ns > len(t)-N0:
       print(' ')
       print('Start simulation too close to end of weather data.')
       print(' ')

    # extract only data for current simulation
    # get an array of all the seconds of the start of the simulation
    t       = D[N0:N0+Ns-1,0]*c                 # Time [s]
    #get the difference in time, this is every five minutes 
    t       = t - t[0]
    dt      = statistics.mean(np.diff(t))       # Sample period data [s]
    if h<dt:
        #h is the stepsize of the model so the stepsize of the sample should be larger then the stepsize of the model 
       print(' ') 
       print('Increase ops.h, sample period too small.')
       print(' ')

    # new sample period p times the original sample rate
    p       = int(np.floor(1/(dt/h)))
    # print()
    rad     = D[N0:N0+Ns+p*Np,1]          # Outdoor Global radiation [W m^{-2}]
    tempOut = D[N0:N0+Ns+p*Np,2]          # Outdoor temperature [°C]
    co2Out  = D[N0:N0+Ns+p*Np,5]          # Outdoor CO2 concentration [ppm]
    co2Out  = co2ppm2dens(tempOut,co2Out) # Outdoor CO2 concentration [kg/m3]
    vapOut  = D[N0:N0+Ns+p*Np,3]          # Outdoor relative humidity [#]
    vapOut  = rh2vaporDens(tempOut,vapOut)# Outdoor humidity [kg/m3]

    # model: d(0) = rad, d(1) = co2Out, d(2) = tempOut, d(3) = vapOut
    d0              = np.array([rad[0], co2Out[0], tempOut[0], vapOut[0]])
    #calculate the amount of samples
    ns              = int(np.ceil(len(rad)/p))
    d               = np.zeros((nd,ns))

    d[0,:]          = d0[0] + signal.resample(rad-d0[0],ns)
    d[1,:]          = d0[1] + signal.resample(co2Out-d0[1],ns)
    d[2,:]          = d0[2] + signal.resample(tempOut-d0[2],ns)
    d[3,:]          = d0[3] + signal.resample(vapOut-d0[3],ns)

    d[0,d[0,:]<0]   = 0
    return d.T

# load_disturbances(c, L, h, nd, Np, weather_data_dir):

def DefineParameters():
    # Model parameters
    p={}

    # parameter 					    description 									[unit] 					nominal value
    p["satH2O1"] = 9348                 # saturation water vapour parameter 			[J m^{-3}] 				9348
    p["satH2O2"] = 17.4                 # saturation water vapour parameter 			[-] 					17.4
    p["satH2O3"] = 239                  # saturation water vapour parameter 			[°C] 					239
    p["satH2O4"] = 10998                # saturation water vapour parameter 			[J m^{-3}] 				10998
    p["R"] = 8314                       # ideal gas constant 							[J K^{-1} kmol^{-1}] 	8314
    p["T"] = 273.15                     # conversion from C to K 						[K] 					273.15
    p["M_CO2"] = 44.01e-3               # molar mass of CO2                             [kg mol^-{1}]           44.01e-3
    p["P"] = 101325                     # pressure (assumed to be 1 atm)                [Pa]                    101325

    p["leak"] = 0.75e-4 				# ventilation leakage through the cover 		[m s^{-1}] 				0.75e-4
    p["CO2cap"] = 4.1 				    # CO2 capacity of the greenhouse 				[m^3{air} m^{-2}{gh}]   4.1
    p["H2Ocap"] = 4.1 				    # Vapor capacity of the greenhouse 				[m^3{air} m^{-2}{gh}]   4.1
    p["aCap"] = 3e4 					# effective heat capacity of the greenhouse air [J m^{-2}{gh} °C^{-1}]  3e4
    p["ventCap"] = 1290 				# heat capacity per volume of greenhouse air 	[J m^{-3}{gh} °C^{-1}]  1290
    p["trans_g_o"] = 6.1 				# overall heat transfer through the cover 		[W m^{-2}{gh} °C^{-1}]  6.1
    p["rad_o_g"] = 0.2   				# heat load coefficient due to solar radiation 	[-] 					0.2

    p["alfaBeta"] = 0.544               # yield factor 									[-] 					0.544
    p["Wc_a"] = 2.65e-7 				# respiration rate 								[s^{-1}] 				2.65e-7
    p["CO2c_a"] = 4.87e-7 			    # respiration coefficient 						[s^{-1}]  				4.87e-7
    p["laiW"] = 53 				   	    # effective canopy surface 						[m^2{leaf} kg^{-1}{dw}] 53
    p["photI0"] = 3.55e-9 			    # light use efficiency 							[kg{CO2} J^{-1}]  		3.55e-9
    p["photCO2_1"]=5.11e-6  			# temperature influence on photosynthesis 		[m s^{-1} °C^{-2}] 		5.11e-6
    p["photCO2_2"]=2.3e-4				# temperature influence on photosynthesis 		[m s^{-1} °C^{-1}] 		2.3e-4
    p["photCO2_3"]=6.29e-4              # temperature influence on photosynthesis 		[m s^{-1}] 				6.29e-4
    p["photGamma"] = 5.2e-5 			# carbon dioxide compensation point 			[kg{CO2} m^{-3}{air}] 	5.2e-5
    p["evap_c_a"] = 3.6e-3 			    # coefficient of leaf-air vapor flow 			[m s^{-1}] 				3.6e-3

    p["dw_fw"] = 22.5                   # dry weigth to fresh weight ratio              [unit]                  22.5
    p["energyCost"] = 6.35e-9/2.20371   # price of energy                               [€ J^{-1}]              6.35e-9 [Dfl J^{-1}] (division by 2.20371 represents currency conversion)
    p["co2Cost"] = 42e-2/2.20371        # price of CO2                                  [€ kg^{-1}{CO2}]        42e-2 [Dfl kg^{-1}{CO2}] (division by 2.20371 represents currency conversion)
    p["productPrice1"] = 1.8/2.20371    # parameter for price of product                [€ m^{-2}{gh}]          1.8 [Dfl kg^{-1}{gh}] (division by 2.20371 represents currency conversion)
    p["productPrice2"] = 16/2.20371     # parameter for price of product                [€ kg^{-1}{gh} m^{-2}{gh}] 16 (division by 2.20371 represents currency conversion)

    # Perturbation parameters
    #  parameter 					    description 									[unit] 					nominal value
    # p["c"].photo = 1                  # perturbation of gross photosynthesis          [-]                     1
    # p["c"].grossPhot = 1              # perturbation of net photosynthesis (bad name!)[-]                     1
    # p["c"].resp = 1                   # perturbation of respiration rate              [-]                     1
    # p["c"].transp = 1                 # perturbation of transpiration rate            [-]                     1
    # p["c"].rad = 1                    # perturbation of outdoor global radiation      [-]                     1
    # p["c"].rhAdd = 0                  # perturbation of outdoor relative humidity     [#]                     0
    # p["c"].tempAdd = 0                # perturbation of outdoor temperature           [°C]                    0
    # p["c"].co2 = 1                    # perturbation of outdoor CO2 concentration     [-]                     1

    p["lue"] = 7.5e-8
    p["heatLoss"] = 1
    p["heatEff"] = 0.1
    p["gasPrice"] = 4.55e-4
    p["lettucePrice"] = 136.4
    p["heatMin"] = 0
    p["heatMax"] = 100
    
    # max operator
    return p

if __name__ == "__main__":
    p = DefineParameters()
    temp = [10,20,30,40]
    for t in temp:
        print(co2ppm2dens(t, 1.6))