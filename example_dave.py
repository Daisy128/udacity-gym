import json
import pathlib
import time
import tqdm
from udacity_gym import UdacitySimulator, UdacityGym, UdacityAction
from udacity_gym.agent import PIDUdacityAgent, DaveUdacityAgent, SupervisedAgent
from udacity_gym.agent_callback import LogObservationCallback

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

    log_observation_callback = LogObservationCallback(log_directory)

    # agent = DaveUdacityAgent(
    #     checkpoint_path="dave2.ckpt",
    #     before_action_callbacks=[],
    #     after_action_callbacks=[log_observation_callback]
    # )
    agent = PIDUdacityAgent()

    # Interacting with the gym environment
    for _ in tqdm.tqdm(range(2000)):
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
