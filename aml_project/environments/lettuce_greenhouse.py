"""
Greenhouse dyanmics modelled in OpenAI Gym environment.
The controller for this environment control the valves of the greenhouse.
That regulates amount of heating (W/m2) and CO_2 into the greenhouse.

For now used modelled as a temperature reference tracking problem.
""" 
import numpy as np

from copy import copy
import matplotlib.pyplot as plt

from aml_project.common.helper_functions import DefineParameters, co2dens2ppm, weather2ppmrh, vaporDens2rh, denormalise_array, load_disturbances
from aml_project.common.plot_gh_variables import create_trajectory_figure, plot_trajectory

import gymnasium as gym
from gymnasium import spaces


class LettuceGreenhouse(gym.Env):
    """
    This class implements the original lettuce greenhouse model by Eldert van Henten (1994),
    in a gymnasium environment. Such that off-the-shelf RL libraries can be applied to control this model.
    It solves the model equations using the numerical Runge-Kutta-4 method.

    Args:
        weather_data_dir    -- path to the weather data
        nx                  -- number of states
        ny                  -- number of measurements
        nd                  -- number of weather variables
        nu                  -- number of control inputs
        control_rate        -- control rate of the system in minutes
        c                   -- number of seconds in a day
        n_days              -- number of simulation days
        Np                  -- number of future weather predictions to use
        start_day           -- starting day of the simulation
        reward_coefs        -- coefficients for the reward function
        penalty_coefs       -- coefficients for the penalty function
    """

    def __init__(self,
        weather_data_dir: str,# path to the weather data
        nx: int = 4,                # number of greenhouse states
        ny: int = 4,                # number of greenhouse measurements
        nd: int = 4,                # number of disturbance (weather variables)
        nu: int = 3,                # number of control inputs
        control_rate: int = 15,     # control rate in minutes
        c: int = 86400,             # conversion to seconds
        n_days: int = 2,             # simulation days
        Np: int = 20,               # number of future predictions (20 == 5hrs)
        start_day: int = 40,         # start day of simulation
        var_weather: bool = False,    # ALWAYS FALSE when postprocessing
        noise: bool = False,    # whether to add noise to the weather predictions
        reward_coefs=np.ones(2),    # coefficients for reward function
        penalty_coefs=np.ones(6)    # coefficients for penalty function
        ):
        super(LettuceGreenhouse, self).__init__()

        # action and observation spaces
        self.action_space = spaces.Box(low=-1*np.ones(nu, dtype=np.float32), high=np.ones(nu, dtype=np.float32))
        # observation space is matrix of (4, Np+1), containing Np weather predictions and the current indoor climate observation
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(ny + nd*Np,))

        # min and max measurements of the environment
        self.obs_low = np.array([0., 0., 10., 0.], dtype=np.float32)
        self.obs_high = np.array([300., 1.6, 25., 80.], dtype=np.float32)

        # min and max actions
        self.min_action = np.array([0.,0.,0.], dtype=np.float32) 
        self.max_action = np.array([1.2
                                    , 7.5
                                    , 150.], dtype=np.float32)

        # previous action and max change in action allowed
        self.prev_action = self.initialise_action()
        self.delta_action = self.max_action/10
        self.noise = noise
        self.p = DefineParameters()
        self.var_weather = var_weather
        self.weather_data_dir = weather_data_dir

        # Number of timestep into prediction horizon
        self.Np = Np
        self.ny = ny
        self.nx = nx
        self.nd = nd

        # simulation parameters
        self.h: int = control_rate*60    # is stepsize in seconds
        self.c: int = c                  # conversion between days and seconds               
        self.n_days: int = n_days
        self.L: int = n_days*c           
        self.N: int = self.L//self.h     # number of steps to take during episode
        self.start_day: int = start_day    # 

        # load weather predictions [currently deterministic]
        self.d = load_disturbances(c, self.L, self.h , self.nd, self.Np, self.start_day, weather_data_dir)

        self.prev_actions = np.zeros((self.N, self.action_space.shape[-1]))
        self.reward_coefs = reward_coefs
        self.penalty_coefs = penalty_coefs

    def seed(self, seed):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_profit(self):
        return self.profit

    def get_revenue(self):
        return self.revenue
    
    def get_co2_cost(self):
        return self.co2_cost
    
    def get_heating_cost(self):
        return self.heating_cost

    def get_penalty(self):
        return self.penalty

    def step(self, action: np.ndarray):
        """
        Step function that simulates one timestep into the future given input action.

        Args:
            actions -- normalised action [array]

        Return:
            observation -- array consisting of four variables
            reward      -- immediate reward of the environment
            terminated  -- whether environment is in a terminal state
            truncated   -- whether the episode was truncated, always false
            info        -- additional information
        """
        # denormalise action
        action = denormalise_array(action, np.clip(self.prev_action-self.delta_action, self.min_action, None), np.clip(self.prev_action+self.delta_action, None, self.max_action))

        # transition state next state given action and observe environment
        self.state = self.f(action, self.d[self.timestep])
        y = self.g()
        self.timestep += 1

        # terminate if state is terminal or crops died
        if self.terminal_state():
            self.terminated = True

        reward = self.reward_function(y[:], action)


        self.prev_action = copy(action)
        self.prev_actions[self.timestep-1] += copy(action)
        self.prev_yield = copy(y[0]) 

        info = self.get_info(action)

        return (
            self.get_obs(y),
            reward, 
            self.terminated, 
            False,
            info
            )

    def get_obs(self, y):
        """
        Generates the observation given the greenhouse measurements and weather measurements.
        Includes a fixed number weather predictions into the future.

        Args:
            y   - Greenhouse measurements
        """
        predictions = self.d[self.timestep:self.timestep + self.Np, :]

        if self.noise:
            noise = np.random.uniform(-0.05, 0.05, predictions.shape) * predictions
            
            predictions += noise
        
        return np.concatenate((y, predictions.flatten()), axis=0)

    def get_info(self, action):
        """
        Get information about the current state of the environment.

        Args:
            action (int): The action taken in the current timestep.

        Returns:
            dict: A dictionary containing the following information:
                - "action": The action taken in the current timestep.
                - "profit": A list of profits for each timestep.
                - "revenue": A list of revenues for each timestep.
                - "co2_cost": A list of CO2 costs for each timestep.
                - "heating_cost": A list of heating costs for each timestep.
                - "penalty": A list of penalties for each timestep.
                - "timestep": The current timestep.
        """
        self.revenues[self.timestep-1] += self.revenue
        self.co2_costs[self.timestep-1] += self.co2_cost
        self.heating_costs[self.timestep-1] += self.heating_cost
        self.profits[self.timestep-1] += self.revenue - self.co2_cost - self.heating_cost
        self.penalties[self.timestep-1] += self.penalty

        info = {
            "action": action,
            "profit": self.profits,
            "revenue": self.revenues,
            "co2_cost": self.co2_costs,
            "heating_cost": self.heating_costs,
            "penalty": self.penalties,
            "timestep": self.timestep
        }
        return info


    def terminal_state(self)-> bool:
        """
        Function that checks whether the environment is in a terminal state.
        If the crop is dead, the temperature is too high or we reached the end of the simulation, the environment is in a terminal state.
        """
        if self.state[0] < 0 or self.timestep >= self.N:
            #print(f"terminal state, timestep: {self.timestep}, crop: {self.state[0]}, temp: {self.state[2]}")
            return True
        return False

    def initialise_action(self):
        """
        Sets the first previous action. Important when generating random trajectories.
        Starts at doing nothing for now.
        """
        return denormalise_array(-np.ones(self.action_space.shape[-1]), self.min_action, self.max_action)

    def reward_function(self, y, action):
        # TODO: add the reward function

        # State Variables
        # y[0] = Dry weight (y1 in paper), y[1] = CO2 ppm (y2), y[2] = Temp C (y3)
        current_weight = y[0]
        co2_ppm = y[1]
        temp_c = y[2]

        # Action Variables
        # action[0] = CO2 dosing, action[1] = Ventilation, action[2] = Heating (all denormalised)
        co2_dosing = action[0]
        ventilation = action[1]
        heating = action[2]

        # Dynamic Constraint Bounds 
        # These are the optimal ranges for temperature and CO2 concentration in the greenhouse.
        T_min = 15.0 # Change so it adjust day/night cycle
        T_max = 20.0
        CO2_min = 800.0
        CO2_max = 1000.0

        # Incremental Growth
        incremental_growth = current_weight - self.prev_yield

        # Climate Constraints (Equation 13) 
        # These are quadratic penalties for being outside the optimal range of CO2 and temperature, and a small reward for being within the optimal range.

        # Coefficients for the reward function
        c_r1, c_r_co2_2 = self.reward_coefs
        c_r_u1, c_r_u2, c_r_u3, c_r_co2_1, c_r_T1, c_r_T2 = self.penalty_coefs

        # CO2 Reward/Penalty
        if co2_ppm < CO2_min:
            r_co2 = -c_r_co2_1 * (co2_ppm - CO2_min)**2
        elif co2_ppm > CO2_max:
            r_co2 = -c_r_co2_1 * (co2_ppm - CO2_max)**2
        else:
            r_co2 = c_r_co2_2  # Small reward for staying in bounds
            
        # Temperature Penalty
        if temp_c < T_min:
            r_T = -c_r_T1 * (temp_c - T_min)**2
        elif temp_c > T_max:
            r_T = -c_r_T2 * (temp_c - T_max)**2
        else:
            r_T = 0.0
            
        self.revenue = float(c_r1 * incremental_growth)
        self.co2_cost = float(c_r_u1 * co2_dosing)
        self.heating_cost = float(c_r_u3 * heating)
        self.penalty = float(r_co2 + r_T - (c_r_u2 * ventilation))
        
        # Final Reward (Equation 12)
        reward = self.revenue + r_co2 + r_T - (self.co2_cost + (c_r_u2 * ventilation) + self.heating_cost)
          
        return float(reward)

    def constraint_penalty(self, obs):
        """
        Function that computes the absolute penalties for violating system constraints.

        Args:
            obs (np.ndarray): The observation array representing the current state of the system.

        Returns:
            penalty (np.ndarray): The absolute penalties for violating system constraints.
        """
        lowerbound = self.obs_low[1:] - obs[1:]
        lowerbound[lowerbound < 0] = 0
        upperbound = obs[1:] - self.obs_high[1:]
        upperbound[upperbound < 0] = 0
        return lowerbound + upperbound

    def reset(self, seed=10):
        """
        Resets environment to starting state.
        Args:
            seed    -- random seed
        Returns:
            observation -- environment state
        """
        super().reset(seed=seed)
        self.timestep = 0        
        self.terminated = False
        self.prev_action = self.initialise_action()
        self.state = np.array([0.0035, 1e-3, 15, 0.008])

        y = self.g()

        self.prev_yield = copy(y[0])
        self.profit = 0
        self.revenue = 0
        self.co2_cost = 0
        self.heating_cost = 0
        self.penalty = 0
        
        if self.var_weather:
            self.start_day = np.random.randint(0, 283)
            self.d = load_disturbances(self.c, self.L, self.h , self.nd, self.Np, self.start_day, self.weather_data_dir)
        else:
            self.d = load_disturbances(self.c, self.L, self.h , self.nd, self.Np, self.start_day, self.weather_data_dir)
        self.profits = np.zeros((self.N, ))
        self.revenues = np.zeros((self.N, ))
        self.co2_costs = np.zeros((self.N, ))
        self.heating_costs = np.zeros((self.N, ))
        self.penalties = np.zeros((self.N, len(self.penalty_coefs)))

        return self.get_obs(y), self.get_info(self.prev_action)

    def close(self):
        return

    def f(self, action, d):
        """
        Finite difference function.
        Computes next state using Runge-Kutta-4 method.
        Args:
            action  --  control variables
            d       --  disturbances of the system,(weather variables)
        Returns:
            state   --  next state of the system
        """
        # finite differencing method to compute new state variables
        k1 = self.F(self.state, action, d, self.p)
        k2 = self.F(self.state+self.h/2 *k1, action, d, self.p)
        k3 = self.F(self.state+self.h/2 *k2, action, d, self.p)
        k4 = self.F(self.state+self.h *k3, action, d, self.p)
        self.state += self.h/6*(k1+ 2*k2 + 2*k3 + k4)
        return self.state

    def g(self):
        """
        Function that 'measures' greenhouse variables.
        Mainly converts the model's state to more human readable metrics.

        Returns:
            y   -- greenhouse measurements.
        """
        y = np.array([1e3*self.state[0],
                1e-3*co2dens2ppm(self.state[2],self.state[1]),
                self.state[2],
                vaporDens2rh(self.state[2], self.state[3])], dtype=np.float32)
        return y

    def F(self, x, u, d, p):
        """
        Function to compute change of x variables using the differential equation.

        Args:
            x   --   state variables
            u   --   control variables
            d   --   (weather) disturbances
            p   --   parameters of the model

        returns:
            delta x --   change of state variables
        """
        ki =  np.array([
            p["alfaBeta"]*(
            (1-np.exp(-p["laiW"] * x[0])) * p["photI0"] * d[0] *
            (-p["photCO2_1"] * x[2]**2 + p["photCO2_2"] * x[2] - p["photCO2_3"]) * (x[1] - p["photGamma"]) 
            / (p["photI0"] * d[0] + (-p["photCO2_1"] * x[2]**2 + p["photCO2_2"] * x[2] - p["photCO2_3"]) * (x[1] - p["photGamma"])))
            - p["Wc_a"] * x[0] * 2**(0.1 * x[2] - 2.5)
            ,

            1 / p["CO2cap"] * (
            -((1 - np.exp(-p["laiW"] * x[0])) * p["photI0"] * d[0] *
            (-p["photCO2_1"] * x[2]**2 + p["photCO2_2"] * x[2] - p["photCO2_3"]) * (x[1] - p["photGamma"])
            / (p["photI0"] * d[0] + (-p["photCO2_1"] * x[2]**2 + p["photCO2_2"] * x[2] - p["photCO2_3"]) * (x[1] - p["photGamma"])))
            + p["CO2c_a"] * x[0] * 2**(0.1 * x[2] - 2.5) + u[0]/1e6 - (u[1] / 1e3 + p["leak"]) * (x[1] - d[1])
            ),

            1/p["aCap"] * (
            u[2] - (p["ventCap"] * u[1] / 1e3 + p["trans_g_o"]) * (x[2] - d[2]) + p["rad_o_g"] * d[0]
            ),

            1/p["H2Ocap"] * ((1 - np.exp(-p["laiW"] * x[0])) * p["evap_c_a"] * (p["satH2O1"]/(p["R"]*(x[2]+p["T"]))*
            np.exp(p["satH2O2"] * x[2] / (x[2] + p["satH2O3"])) - x[3]) - (u[1]/1e3 + p["leak"]) * (x[3] - d[3]))]
            )
        return ki

    def generate_trajectory(self):
            """
            Generates a trajectory for the greenhouse environment from the starting state.
            The trajectory is generated by randomly taking steps in the environment until termination.

            Returns:
                y: Measurements, weather, and action trajectory.
                d: Weather trajectory.
                u: Action trajectory.
            """
            # Initialize arrays to store measurements, actions, and rewards
            y = np.zeros((self.N+1, self.ny))
            u = np.zeros((self.N, self.action_space.shape[0]))
            rewards = np.zeros(self.N)

            # Set initial state and action
            y[0] = self.g()
            self.prev_action = np.zeros(self.action_space.shape[0])

            # Initialize loop variables
            k = 0
            terminated = False

            # Generate trajectory until termination
            while not terminated:
                # Randomly sample an action from the action space
                action = self.action_space.sample()

                # Take a step in the environment
                observation, reward, terminated, truncated, info = self.step(action)

                # Store the previous action
                u[k] = self.prev_action

                # Store the observation as a measurement
                y[k+1] = observation[:self.ny]

                # Accumulate the reward
                rewards[k] += reward

                # Increment the step counter
                k += 1

            # Return the generated trajectory
            return y[:self.N], weather2ppmrh(self.d[:self.N]), u[:self.N]

if __name__ == "__main__":
    nx = 4
    ny = 4
    nu = 3
    nd = 4

    #  simulation parameters
    c = 86400
    # N days in simulation (max = 46 weeks)
    n_days = 2                   
    # sample period in seconds: #minutes*60 every 900 seconds we sample
    h = 15*60                   
    # final time simulation
    L = n_days*c                 
    # initial time vector
    t = np.arange(0, L, h) 
    # number of samples in initial time vector
    N = len(t)                  
    Np = 0
    # disturbances of the weather loaded in via matlab
    # consists of 4 variables
    # 1) incoming radiation I_o [W.m^-2]
    # 2) outside carbon dioxide concentration C_{CO2_o} [kg.m^-3]
    # 3) outside Temperature outside T_o
    # 4) outside humidity content C_{H2O}_o
    # print(d.shape)
    # 1) dry matter the content of the lettuce W [kg.m^2]
    # 2) indoor carbon dioxide concentration C_{CO_2}_a [kg.m^-3]
    # 3) indoor air temperature T_a in celcius degrees
    # 4) indoor humidity C_{H2O}_a [kg.m^-3]
    x = np.zeros((nx, N+1))

    # 4 measurements variables
    # 1) W in [g.m^-2]
    # 2) indoor C_{CO2} in ppm
    # 3) indoor temperature T_a in celcius
    # 4) relative humidity in %
    y = np.zeros((ny, N))

    # three controllable variable
    # 1) supply rate of carbon dioxide u_{CO2} in [mg.m^-2.s^-2]
    # 2) ventialation rate through the vents in [mm.s^-1]
    # 3) energy supply by heating the system u_q in [W.m^-2]
    u = np.zeros((nu, N+1))
    weather_direction = 'environments/weather/outdoorWeatherWurGlas2014.mat'
    penalty_coefs= np.array([0.5, 0.5, 0.5])
    reward_coefs = np.array([1, 0.1, .01, .001])

    env = LettuceGreenhouse(weather_direction, penalty_coefs=penalty_coefs, reward_coefs=reward_coefs)
    env.reset()
    y, d, u = env.generate_trajectory()
    nvars = 11

    # since weather variables are in raw metrics
    # we convert both here CO2 concentration to parts per million (ppm) and vapor pressure deficit to relative humidity
    d = weather2ppmrh(d)

    # plot the resulting trajectory
    n_per_day = int(24*3600/env.h) # control frequency per day
    label= 'Random policy'
    trajectory = np.concatenate((y, d, u), axis=1)
    fig, axes = create_trajectory_figure(nvars, env.L, env.h, env.c, None, None)
    fig, axes = plot_trajectory(fig, axes, trajectory, env.L, env.h, env.c, 0, n_days, n_per_day, label)
    plt.show()