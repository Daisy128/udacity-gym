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
    simulator_exe_path = "/home/jiaqq/Documents/builds_v2/udacity.x86_64"

    # 4 track variable settings
    track = "mountain"
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
        #print("Waiting for environment to set up...")
        time.sleep(1)

    model_path = f"./models/{Track(track).name}/track3-dave2-mc-final.h5"#014 final
    #model_path = f"./models/{Track(track).name}-dave2-final copy.h5"

    #print(model_path)

    #model_path = "./models/track1-dave2-20240907_230759-final.h5"  
    log_observation_callback = LogObservationCallback(log_directory)
    agent = SupervisedAgent(model_path=model_path,
                            max_speed=15,
                            min_speed=6,
                            predict_throttle=False)

    # Interacting with the gym environment
    # tqdm.tqdm() creates a progress bar, consists of multiple steps where each step represents an action taken by the agent
    # 2000: agent executes 2000 steps in total
    for _ in tqdm.tqdm(range(6000)): 
        action = agent(observation) # agent 根据当前的环境状态或观察值（observation）来选择一个动作
        last_observation = observation
        # 强化学习, 输入一个动作，返回
        # observation - 采取该动作后的环境状态; rewards - 采取该动作获得的奖励; 
        # terminated - 当前 episode 是否结束, boolean; truncated - 是否因为某些条件（如时间限制）而提前终止 episode; 
        # info - extra info, not compulsory
        observation, reward, terminated, truncated, info = env.step(action)

        # 代码持续观察环境的变化，直到时间更新为止, 确保每一步操作之间的时间确实发生变化
        while observation.time == last_observation.time: # time equals, 环境处于同一个时间点
            observation = env.observe() # inside while loop, keep observing
            time.sleep(0.005)

    if info:
        json.dump(info, open(log_directory.joinpath("info.json"), "w"))

    log_observation_callback.save()
    simulator.close()
    env.close()
    #print("Experiment concluded.")


