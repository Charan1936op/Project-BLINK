# In src/run.py

import os
import time
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import traci
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure

print("\n--- RUNNING FROM CLEAN SCRIPT: run.py ---")

# ==============================================================================
#  ENVIRONMENT DEFINITION
# ==============================================================================
class MultiIntersectionEnv(gym.Env):
    def __init__(self, sumo_config, use_gui=False):
        super(MultiIntersectionEnv, self).__init__()
        self.sumo_binary = "sumo-gui" if use_gui else "sumo"
        self.sumo_config = sumo_config
        self.max_cars = 50
        self.tls_ids = []
        self.green_phases = {}
        obs_dim = 4
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def _start_sumo(self):
        sumo_cmd = [self.sumo_binary, "-c", self.sumo_config, "--start", "--quit-on-end", "--no-warnings", "true"]
        traci.start(sumo_cmd)
        self.tls_ids = traci.trafficlight.getIDList()
        if not self.tls_ids:
            raise Exception("No traffic lights found.")
        for tls in self.tls_ids:
            program = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)[0]
            # green phases: phases without 'y' (yellow) in state
            self.green_phases[tls] = [i for i, p in enumerate(program.phases) if 'y' not in p.state.lower()]

    def reset(self, seed=None, options=None):
        if traci.isLoaded():
            traci.close()
        self._start_sumo()
        return {tls: self._get_observation(tls) for tls in self.tls_ids}, {}

    def step(self, actions):
        rewards, new_obs, dones, truncated, infos = {}, {}, {}, {}, {}
        for tls in self.tls_ids:
            act = int(actions.get(tls, 0))
            # if action 1 -> try to switch to next green phase (simple policy)
            if act == 1 and len(self.green_phases.get(tls, [])) > 1:
                curr_phase = traci.trafficlight.getPhase(tls)
                greens = self.green_phases[tls]
                try:
                    idx = greens.index(curr_phase)
                    next_green = greens[(idx + 1) % len(greens)]
                except ValueError:
                    next_green = greens[0]
                traci.trafficlight.setPhase(tls, int(next_green))

        traci.simulationStep()

        for tls in self.tls_ids:
            new_obs[tls] = self._get_observation(tls)
            rewards[tls] = self._get_reward(tls)
            dones[tls] = False
            truncated[tls] = False
            infos[tls] = {}
            
        return new_obs, rewards, dones, truncated, infos

    def _get_observation(self, tls):
        lanes = list(set(traci.trafficlight.getControlledLanes(tls)))
        counts = [traci.lane.getLastStepVehicleNumber(l) for l in lanes]
        waits = [traci.lane.getWaitingTime(l) for l in lanes]
        obs = np.array([
            np.mean(counts) / self.max_cars if counts else 0.0,
            np.max(counts) / self.max_cars if counts else 0.0,
            np.mean(waits) / 100.0 if waits else 0.0,
            np.max(waits) / 100.0 if waits else 0.0
        ], dtype=np.float32)
        return np.nan_to_num(obs)

    def _get_reward(self, tls):
        lanes = list(set(traci.trafficlight.getControlledLanes(tls)))
        return -sum(traci.lane.getWaitingTime(l) for l in lanes) / 100.0

    def close(self):
        if traci.isLoaded():
            traci.close()


# ==============================================================================
#  TRAINING SCRIPT
# ==============================================================================
if __name__ == "__main__":
    SUMO_CONFIG = "sumo_files/multi.sumocfg"
    MODEL_DIR = "models"
    TOTAL_TIMESTEPS = 250000

    os.makedirs(MODEL_DIR, exist_ok=True)

    # STEP 1: Define environment properties without starting SUMO.
    OBS_SPACE = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
    ACT_SPACE = spaces.Discrete(2)

    class DummyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = OBS_SPACE
            self.action_space = ACT_SPACE

        def step(self, action):
            # return dummy next_obs, reward, terminated, truncated, info
            return np.zeros(OBS_SPACE.shape, dtype=np.float32), 0.0, False, False, {}

        def reset(self, seed=None, options=None):
            return np.zeros(OBS_SPACE.shape, dtype=np.float32), {}

    init_env = DummyEnv()

    # Do a quick, temporary run of SUMO to get the agent IDs.
    print("Finding traffic light IDs...")
    temp_env = MultiIntersectionEnv(SUMO_CONFIG)
    _, _ = temp_env.reset()
    tls_ids = temp_env.tls_ids
    temp_env.close()
    print(f"Found agents: {tls_ids}")

    # STEP 2: Fully create and set up all models BEFORE starting the main simulation.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    models = {}
    for tls in tls_ids:
        print(f"Initializing model for agent: {tls}")
        models[tls] = DQN(
            policy="MlpPolicy",
            env=init_env,
            learning_rate=1e-4,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.95,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            verbose=0,
            device=device,
        )

    print("Setting up models and loggers...")
    for tls in tls_ids:
        log_dir = os.path.join("logs", tls)
        os.makedirs(log_dir, exist_ok=True)
        # Try tensorboard first; if unavailable, fall back to stdout+csv
        try:
            logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
        except Exception:
            logger = configure(log_dir, ["stdout", "csv"])
        models[tls].set_logger(logger)
        # ensure internal structures (policy, replay buffer, optimizer) are created
        models[tls]._setup_model()
    print("All models successfully set up.")

    # STEP 3: NOW start the REAL simulation for the training loop.
    real_env = MultiIntersectionEnv(SUMO_CONFIG, use_gui=False)
    obs, _ = real_env.reset()

    print("Starting training loop...")
    start_time = time.time()
    for step in range(TOTAL_TIMESTEPS):
        actions = {}
        for tls in tls_ids:
            act, _ = models[tls].predict(obs[tls], deterministic=False)
            actions[tls] = int(act)

        new_obs, rewards, dones, truncated, infos = real_env.step(actions)

        for tls in tls_ids:
            # Add to replay buffer (SB3 expects arrays) including infos argument.
            models[tls].replay_buffer.add(
                obs[tls],
                new_obs[tls],
                np.array([actions[tls]]),
                np.array([rewards[tls]]),
                np.array([dones[tls]]),
                [infos[tls]]
            )
            if step > models[tls].learning_starts:
                models[tls].train(gradient_steps=1)

        obs = new_obs
        if any(dones.values()):
            obs, _ = real_env.reset()

        if step > 0 and step % 5000 == 0:
            elapsed_time = time.time() - start_time
            sps = step / elapsed_time if elapsed_time > 0 else 0.0
            remaining = TOTAL_TIMESTEPS - step
            eta = remaining / sps if sps > 0 else float("inf")
            hrs, rem = divmod(eta, 3600)
            mins, secs = divmod(rem, 60)
            print(f"Step {step}/{TOTAL_TIMESTEPS} | {sps:.2f} steps/s | ETA: {int(hrs)}h {int(mins)}m")
            for tls, model in models.items():
                model.save(os.path.join(MODEL_DIR, f"marl_dqn_{tls}.zip"))

    print("Training finished. Saving final models...")
    for tls, model in models.items():
        model.save(os.path.join(MODEL_DIR, f"marl_dqn_{tls}.zip"))

    real_env.close()
    print("✅ MARL training completed successfully.")