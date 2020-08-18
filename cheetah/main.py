### append path ###
import sys
import os
PATH = os.path.dirname(os.path.abspath(__file__))
PATH = ('/').join(PATH.split('/')[:-1])
sys.path.append(PATH)
###################

#from agent import Agent
from c_module.agent import Agent
from env.env import Env

import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import time

def main():
    env = Env(enable_draw=True, base_fix=False)
    agent = Agent(env)

    delta_time = 0.025
    time_horizon = 10
    com_pos = np.array([0.0, 0, 0.25])
    rpy = np.zeros(3)
    com_vel = np.zeros(3)
    base_ang_vel = np.zeros(3)
    target_x = np.concatenate([com_pos, rpy, com_vel, base_ang_vel])
    target_x = target_x.reshape((-1, 1))
    target_u = np.array([0, 0, env.model.mass*0.25*9.8]*4).reshape((12, 1))
    init_u_list = np.array([target_u for i in range(time_horizon)])

    temp_length = int(0.3/delta_time)
    temp_contact_phi_list = [[0,1,1,0]]*temp_length+[[1,1,1,1]]*temp_length+[[1,0,0,1]]*temp_length+[[1,1,1,1]]*temp_length
    total_contact_phi_list = np.array([[1,1,1,1]]*temp_length+temp_contact_phi_list*1000)

    state = env.reset()
    t = 0
    last_t = 0
    while t<100:

        if last_t == 0 or t-last_t >= delta_time:
            last_t = t
            com_pos = env.model.com_pos
            print(com_pos)
            rpy = env.model.base_rpy
            com_vel = env.model.base_vel
            base_ang_vel = np.matmul(env.model.base_rot.T, env.model.base_ang_vel)
            init_x = np.concatenate([com_pos, rpy, com_vel, base_ang_vel])
            init_x = init_x.reshape((-1, 1))

            delta_time_list = np.array([delta_time]*time_horizon)
            foot_pos_list = np.array([env.model.foot_pos_list for i in range(time_horizon + 1)])
            contact_phi_list = total_contact_phi_list[:time_horizon+1]
            total_contact_phi_list = total_contact_phi_list[1:]

            target_x_list = np.array([target_x for i in range(time_horizon + 1)])
            target_u_list = np.array([target_u for i in range(time_horizon)])

            action, u_list  = agent.get_action(init_x, init_u_list, delta_time_list, foot_pos_list, contact_phi_list, target_x_list, target_u_list)
            init_u_list = deepcopy(u_list)
            for leg_idx in range(4):
                if contact_phi_list[0, leg_idx] == 0.0:
                    action[leg_idx*3:(leg_idx+1)*3] = [0, 0, -3.0]

        state = env.step(action, contact_phi_list[0, :])

        t += env.time_step

if __name__ == "__main__":
    main()
