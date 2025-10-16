# In src/marl_env.py

import os
import numpy as np
import traci
import gymnasium as gym
from gymnasium import spaces

class MultiIntersectionEnv(gym.Env):
    """
    Multi-agent SUMO environment where each traffic light is an agent.
    This version dynamically finds the green phases for each traffic light.
    """

    def __init__(self, sumo_config, use_gui=False):
        super(MultiIntersectionEnv, self).__init__()

        self.sumo_binary = "sumo-gui" if use_gui else "sumo"
        self.sumo_config = sumo_config
        self.max_cars = 50
        
        self.tls_ids = []
        # --- FIX: Dictionary to store the green phase indices for each TLS ---
        self.green_phases = {}

        obs_dim = 4
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0 = keep, 1 = switch

    def _start_sumo(self):
        """Starts a new SUMO simulation."""
        sumo_cmd = [self.sumo_binary, "-c", self.sumo_config, "--start", "--quit-on-end", "--no-warnings", "true"]
        traci.start(sumo_cmd)
        self.tls_ids = traci.trafficlight.getIDList()
        if not self.tls_ids:
            raise Exception("No traffic lights found in SUMO network.")
            
        # --- FIX: Dynamically find and store green phases for each traffic light ---
        for tls in self.tls_ids:
            program = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)[0]
            # A "green" phase is a primary phase, typically without any yellow lights ('y')
            self.green_phases[tls] = [
                i for i, phase in enumerate(program.phases) if 'y' not in phase.state.lower()
            ]
            # Ensure we found at least one green phase to prevent errors
            if not self.green_phases[tls]:
                 raise ValueError(f"No green phases found for traffic light {tls}")


    def reset(self, seed=None, options=None):
        """Resets the environment by starting a new SUMO simulation."""
        if traci.isLoaded():
             traci.close()
        self._start_sumo()
        
        obs = {tls: self._get_observation(tls) for tls in self.tls_ids}
        return obs, {}

    def step(self, actions):
        rewards, new_obs, dones, truncated, infos = {}, {}, {}, {}, {}

        for tls in self.tls_ids:
            act = actions.get(tls, 0)

            # --- FIX: Use dynamic phase switching logic ---
            if act == 1 and len(self.green_phases[tls]) > 1:
                current_phase = traci.trafficlight.getPhase(tls)
                
                # Find the next green phase in the cycle
                try:
                    current_green_index = self.green_phases[tls].index(current_phase)
                    next_green_index = (current_green_index + 1) % len(self.green_phases[tls])
                    next_phase = self.green_phases[tls][next_green_index]
                    traci.trafficlight.setPhase(tls, next_phase)
                except ValueError:
                    # If current phase is yellow, it won't be in green_phases.
                    # Default to setting the first green phase.
                    traci.trafficlight.setPhase(tls, self.green_phases[tls][0])

        traci.simulationStep()

        for tls in self.tls_ids:
            obs = self._get_observation(tls)
            reward = self._get_reward(tls)
            new_obs[tls] = obs
            rewards[tls] = float(reward)
            dones[tls] = traci.simulation.getMinExpectedNumber() <= 0
            truncated[tls] = False
            infos[tls] = {}

        return new_obs, rewards, dones, truncated, infos

    def _get_observation(self, tls):
        lane_list = list(set(traci.trafficlight.getControlledLanes(tls)))
        veh_counts = [traci.lane.getLastStepVehicleNumber(l) for l in lane_list]
        wait_times = [traci.lane.getWaitingTime(l) for l in lane_list]

        obs = np.array([
            np.mean(veh_counts) / self.max_cars if veh_counts else 0,
            np.max(veh_counts) / self.max_cars if veh_counts else 0,
            np.mean(wait_times) / 100.0 if wait_times else 0,
            np.max(wait_times) / 100.0 if wait_times else 0,
        ], dtype=np.float32)
        return np.nan_to_num(obs)

    def _get_reward(self, tls):
        lanes = list(set(traci.trafficlight.getControlledLanes(tls)))
        total_wait = sum(traci.lane.getWaitingTime(l) for l in lanes)
        return -total_wait / 100.0

    def close(self):
        """Closes the TraCI connection."""
        if traci.isLoaded():
            traci.close()