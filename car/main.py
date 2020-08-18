from car_env.env import Env
from c_module.agent import Agent

import matplotlib.pyplot as plt
import numpy as np
import time

def main():
    env = Env()
    agent = Agent(env)

    for i in range(100):
        done = False
        state = env.reset()
        agent.reset()
        while not done:
            action = agent.get_action(state, env.hazard_list, env.hazard_radius)
            state, reward, done, info = env.step(action)
            env.render()

if __name__ == "__main__":
    main()
