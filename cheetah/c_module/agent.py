### append path ###
import sys
import os
PATH = os.path.dirname(os.path.abspath(__file__))
PATH = ('/').join(PATH.split('/')[:-2])
if not PATH in sys.path:
    sys.path.append(PATH)
###################

from env.env import Env

from copy import deepcopy
from ctypes import cdll
import numpy as np
import ctypes
import time
import os


class Agent:
    def __init__(self, env=None):
        self.env = env
        self.x_dim = 3*4
        self.u_dim = 4*3
        self.damping_ratio = 1e-8
        self.learning_rate = 1.0 #0.5
        self.max_iteration = 10
    
        self.Qf_mat = 200.0*np.eye(self.x_dim)
        self.Qf_mat = np.matmul(self.Qf_mat.T, self.Qf_mat)
        self.Q_mat = 1.0*np.eye(self.x_dim)
        self.Q_mat = np.matmul(self.Q_mat.T, self.Q_mat) 
        self.R_mat = 0.2*np.eye(self.u_dim)
        self.R_mat = np.matmul(self.R_mat.T, self.R_mat) 

        #for constraint
        self.const_dim = 17
        self.init_lambda_vector = np.zeros((self.const_dim, 1))
        self.init_mu_vector = np.ones((self.const_dim, 1))
        self.mu_scaling_factor = 2.0
        self.max_lambda = 10.0

        ABS_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.CPP_LIB = cdll.LoadLibrary('{}/ALiLQR.so'.format(ABS_PATH))


    def get_action(self, init_x, init_u_list, delta_time_list, foot_pos_list, contact_phi_list, target_x_list, target_u_list):
        x_dim = self.x_dim
        u_dim = self.u_dim
        c_dim = self.const_dim
        damping_ratio = self.damping_ratio
        learning_rate = self.learning_rate
        max_iteration = self.max_iteration
        R_mat = self.R_mat
        Qf_mat = self.Qf_mat
        Q_mat = self.Q_mat
        init_lambda_vector = self.init_lambda_vector
        init_mu_vector = self.init_mu_vector
        mu_scaling_factor = self.mu_scaling_factor
        max_lambda = self.max_lambda
        CPP_LIB = self.CPP_LIB

        time_horizon = len(delta_time_list)
        return_var = np.zeros(u_dim)
        ctype_arr_convert = lambda arr : (ctypes.c_double * len(arr))(*arr)

        def temp_ravel(args): 
            for arg_idx in range(len(args)): 
                args[arg_idx] = np.array(args[arg_idx], dtype=np.float64).ravel() 
            return args 
        [init_x, init_u_list, delta_time_list, foot_pos_list, contact_phi_list, target_x_list, target_u_list, R_mat, Qf_mat, Q_mat, init_lambda_vector, init_mu_vector, return_var] = \
        temp_ravel([init_x, init_u_list, delta_time_list, foot_pos_list, contact_phi_list, target_x_list, target_u_list, R_mat, Qf_mat, Q_mat, init_lambda_vector, init_mu_vector, return_var])

        CPP_LIB.get_action.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double*x_dim), ctypes.POINTER(ctypes.c_double*(u_dim*time_horizon)), ctypes.POINTER(ctypes.c_double*time_horizon), 
                                ctypes.POINTER(ctypes.c_double*(4*3*(time_horizon + 1))), ctypes.POINTER(ctypes.c_double*(4*(time_horizon + 1))), ctypes.POINTER(ctypes.c_double*(x_dim*(time_horizon + 1))),
                                ctypes.POINTER(ctypes.c_double*(u_dim*time_horizon)), ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double*u_dim**2), 
                                ctypes.POINTER(ctypes.c_double*x_dim**2), ctypes.POINTER(ctypes.c_double*x_dim**2), ctypes.POINTER(ctypes.c_double*c_dim), ctypes.POINTER(ctypes.c_double*c_dim), ctypes.c_double,
                                ctypes.c_double, ctypes.POINTER(ctypes.c_double*u_dim))

        cpp_init_x = ctype_arr_convert(init_x)
        cpp_init_u_list = ctype_arr_convert(init_u_list)
        cpp_delta_time_list = ctype_arr_convert(delta_time_list)
        cpp_foot_pos_list = ctype_arr_convert(foot_pos_list)
        cpp_contact_phi_list = ctype_arr_convert(contact_phi_list)
        cpp_target_x_list = ctype_arr_convert(target_x_list)
        cpp_target_u_list = ctype_arr_convert(target_u_list)
        cpp_damping_ratio = ctypes.c_double(damping_ratio)
        cpp_learning_rate = ctypes.c_double(learning_rate)
        cpp_max_iteration = ctypes.c_int(max_iteration)
        cpp_R_mat = ctype_arr_convert(R_mat)
        cpp_Qf_mat = ctype_arr_convert(Qf_mat)
        cpp_Q_mat = ctype_arr_convert(Q_mat)
        cpp_init_lambda_vector = ctype_arr_convert(init_lambda_vector)
        cpp_init_mu_vector = ctype_arr_convert(init_mu_vector)
        cpp_mu_scaling_factor = ctypes.c_double(mu_scaling_factor)
        cpp_max_lambda = ctypes.c_double(max_lambda)
        cpp_return_var = ctype_arr_convert(return_var)

        CPP_LIB.get_action(time_horizon, cpp_init_x, cpp_init_u_list, cpp_delta_time_list, cpp_foot_pos_list, cpp_contact_phi_list, cpp_target_x_list, cpp_target_u_list, cpp_damping_ratio,
                            cpp_learning_rate, cpp_max_iteration, cpp_R_mat, cpp_Qf_mat, cpp_Q_mat, cpp_init_lambda_vector, cpp_init_mu_vector, cpp_mu_scaling_factor, cpp_max_lambda, cpp_return_var)
        action = np.array(cpp_return_var)
        u_list = np.array(cpp_init_u_list).reshape((time_horizon, u_dim))
        return action, u_list


def main():
    env = Env(enable_draw=True, base_fix=False)
    agent = Agent(env)
    
    time_horizon = 10
    com_pos = np.array([0.0, 0, 0.1])
    rpy = np.zeros(3)
    com_vel = np.zeros(3)
    base_ang_vel = np.zeros(3)
    target_x = np.concatenate([com_pos, rpy, com_vel, base_ang_vel])
    target_x = target_x.reshape((-1, 1))
    target_u = np.array([0, 0, env.model.mass*0.25*9.8]*4).reshape((12, 1))
    init_u_list = np.array([target_u for i in range(time_horizon)])

    state = env.reset()
    t = 0
    while t<10:
        com_pos = env.model.com_pos
        rpy = env.model.base_rpy
        com_vel = env.model.base_vel
        base_ang_vel = np.matmul(env.model.base_rot.T, env.model.base_ang_vel)
        init_x = np.concatenate([com_pos, rpy, com_vel, base_ang_vel])
        init_x = init_x.reshape((-1, 1))

        delta_time_list = np.array([0.01]*time_horizon)
        foot_pos_list = np.array([env.model.foot_pos_list for i in range(time_horizon + 1)])
        contact_phi_list = np.array([[1, 1, 1, 1] for i in range(time_horizon + 1)])

        target_x_list = np.array([target_x for i in range(time_horizon + 1)])
        target_u_list = np.array([target_u for i in range(time_horizon)])

        action, u_list  = agent.get_action(init_x, init_u_list, delta_time_list, foot_pos_list, contact_phi_list, target_x_list, target_u_list)
        init_u_list = deepcopy(u_list)

        state = env.step(action)

        #time.sleep(env.time_step)
        t += env.time_step

if __name__ == "__main__":
    main()
