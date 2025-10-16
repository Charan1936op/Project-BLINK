# In src/inference_marl.py (This code is correct, assuming the training script fix)

from src.marl_env import MultiIntersectionEnv
from stable_baselines3 import DQN
import os, time
from traci.exceptions import FatalTraCIError

env = MultiIntersectionEnv("sumo_files/multi.sumocfg", use_gui=True)
obs, _ = env.reset()

agents = {}
for tls in env.tls_ids:
    # This path will now work correctly
    path = f"models/marl_dqn_{tls}.zip"
    if os.path.exists(path):
        agents[tls] = DQN.load(path)
        print(f"✅ Model loaded for {tls}")
    else:
        print(f"⚠️ No model found for {tls}, this agent will not act.")

# Check if any agents were loaded
if not agents:
    print("❌ No models found. Exiting. Please train the models first.")
    exit()

try:
    # Run simulation until it ends
    while True:
        actions = {}
        for tls, agent in agents.items():
            # Ensure obs for this agent is available before predicting
            if tls in obs:
                actions[tls], _ = agent.predict(obs[tls], deterministic=True)
            else:
                actions[tls] = 0  # Default action if obs is missing

        obs, rewards, dones, _, _ = env.step(actions)

        # If the simulation is over for all agents, break the loop
        if all(dones.values()):
            print("Simulation finished.")
            break

        time.sleep(0.1)
except FatalTraCIError:
    print("SUMO simulation closed connection. Exiting gracefully.")
finally:
    env.close()