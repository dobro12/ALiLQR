from c_module.agent import Agent

from scipy.spatial.transform import Rotation as R
from collections import deque
import numpy as np
import safety_gym
import pickle
import time
import sys
import gym

env_name = 'Safexp-CarGoal1-v0'

def train():
    env = gym.make(env_name)
    agent = Agent(env)

    epochs = 100
    
    for epoch in range(epochs):
        env.reset()
        agent.reset()

        done = False
        score = 0
        while not done:
            state = env.data.get_joint_qpos('robot')
            hazard_list = []
            for h_idx in range(len(env.hazards_pos)):
                hazard_list.append((env.hazards_pos[h_idx] - env.goal_pos)[:2])
            hazard_list = np.array(hazard_list)
            hazard_radius = 0.35 #env.hazards_size

            r= R.from_quat(state[-4:])
            angle = np.pi/2 - r.as_rotvec()[0]
            state = np.array([state[0] - env.goal_pos[0], state[1] - env.goal_pos[1], angle]).reshape((3,1))

            action = agent.get_action(state, hazard_list, hazard_radius)

            next_state, reward, done, info = env.step(action)
            env.render()
            score += reward
        print(score)

if len(sys.argv)== 2 and sys.argv[1] == 'test':
    test()
else:
    train()
