import json
import pathlib
import time
import tqdm
import os
from enum import Enum
from udacity_gym import UdacitySimulator, UdacityGym, UdacityAction
from udacity_gym.agent import PIDUdacityAgent, DaveUdacityAgent, SupervisedAgent
from udacity_gym.agent_callback import LogObservationCallback

class Track(Enum):
    track1 = "lake"
    track2 = "jungle"
    track3 = "mountain"

if __name__ == '__main__':

    # Configuration settings
    host = "127.0.0.1"
    port = 4567
    simulator_exe_path = "/home/jiaqq/Documents/Builds/udacity_linux.x86_64"

    # 4 track variable settings
    track = "lake"
    daytime = "day"
    weather = "sunny"
    log_directory = pathlib.Path(f"udacity_dataset_lake_dave/{track}_{weather}_{daytime}")

    # Creating the simulator wrapper
    simulator = UdacitySimulator(
        sim_exe_path=simulator_exe_path,
        host=host,
        port=port,
    )

    # Creating the gym environment
    env = UdacityGym(
        simulator=simulator,
    )
    simulator.start()
    observation, _ = env.reset(track=f"{track}", weather=f"{weather}", daytime=f"{daytime}")

    # Wait for environment to set up
    while not observation or not observation.is_ready():
        observation = env.observe()
        print("Waiting for environment to set up...")
        time.sleep(1)

    model_path = f"./models/{Track(track).name}-dave2-final.h5"
   # model_path = "./models/track1-dave2-20240901_213047-final.h5"  
    log_observation_callback = LogObservationCallback(log_directory)
    agent = SupervisedAgent(model_path=model_path,
                            max_speed=40,
                            min_speed=10,
                            predict_throttle=False)

    # Interacting with the gym environment
    for _ in tqdm.tqdm(range(500)):
        action = agent(observation)
        last_observation = observation
        observation, reward, terminated, truncated, info = env.step(action)

        while observation.time == last_observation.time:
            observation = env.observe()
            time.sleep(0.005)

    if info:
        json.dump(info, open(log_directory.joinpath("info.json"), "w"))

    log_observation_callback.save()
    simulator.close()
    env.close()
    print("Experiment concluded.")


