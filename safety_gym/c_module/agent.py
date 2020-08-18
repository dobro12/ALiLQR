from ctypes import cdll
import numpy as np
import ctypes
import os

TIME_HORIZON = 20
U_DIM = 2
X_DIM = 3
COST_DIM = 8

ABS_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-1])

class Agent:
    def __init__(self, env=None):
        self.x_dim = X_DIM
        self.u_dim = U_DIM
        self.delta_t = 0.01
        self.time_horizon = TIME_HORIZON
        self.damping_ratio = 0.0001
        self.learning_rate = 0.5

        assert U_DIM==self.u_dim and X_DIM==self.x_dim
    
        self.R_mat = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.R_mat = np.matmul(self.R_mat.T, self.R_mat) 
        self.Qf_mat = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 0.0]])
        self.Qf_mat = np.matmul(self.Qf_mat.T, self.Qf_mat)
        self.Q_mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        self.Q_mat = np.matmul(self.Q_mat.T, self.Q_mat) 
        self.target_x = np.array([[0.0], [0.0], [0.0]])

        #init lambda vector, trainig parameter 이므로 저장하고, 로딩하는 식으로 사용해야함.
        self.lambda_vector = np.zeros(COST_DIM)
        self.mu_vector = np.ones(COST_DIM)*10.0

        #init u list
        self.u_list = np.zeros((self.time_horizon, self.u_dim, 1)) #u : 1x1, u_list : N*1*1


        ######################################
        ######## get iLQR.so function ########
        test = cdll.LoadLibrary('{}/ALiLQR.so'.format(ABS_PATH))
        test.get_action.argtypes = (ctypes.POINTER(ctypes.c_double * X_DIM), ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double * U_DIM**2), \
                                    ctypes.POINTER(ctypes.c_double * X_DIM**2), ctypes.POINTER(ctypes.c_double * X_DIM**2), \
                                    ctypes.POINTER(ctypes.c_double * X_DIM), ctypes.POINTER(ctypes.c_double * (U_DIM * TIME_HORIZON)), \
                                    ctypes.POINTER(ctypes.c_double * COST_DIM), ctypes.POINTER(ctypes.c_double * COST_DIM), \
                                    ctypes.POINTER(ctypes.c_double * (2*COST_DIM)), ctypes.c_double)
        test.get_action.restype = ctypes.POINTER(ctypes.c_double * (U_DIM*TIME_HORIZON))
        self.iLQR = test
        self.ctype_arr_convert = lambda arr : (ctypes.c_double * len(arr))(*arr)
        ######## get iLQR.so function ########
        ######################################

    def reset(self):
        #init u list
        self.u_list = np.zeros((self.time_horizon, self.u_dim, 1)) #u : 1x1, u_list : N*1*1

    def get_action(self, init_x, hazard_list, hazard_radius):
        delta_t = self.delta_t
        damping_ratio = self.damping_ratio
        learning_rate = self.learning_rate

        R_mat = self.ctype_arr_convert(self.R_mat.ravel())
        Qf_mat = self.ctype_arr_convert(self.Qf_mat.ravel())
        Q_mat = self.ctype_arr_convert(self.Q_mat.ravel())
        target_x = self.ctype_arr_convert(self.target_x.ravel())
        u_list = self.ctype_arr_convert(self.u_list.ravel())
        init_x = self.ctype_arr_convert(init_x.ravel())
        lambda_vector = self.ctype_arr_convert(self.lambda_vector.ravel())
        mu_vector = self.ctype_arr_convert(self.mu_vector.ravel())
        hazard_list = self.ctype_arr_convert(hazard_list.ravel())

        result = self.iLQR.get_action(init_x, delta_t, damping_ratio, learning_rate, \
                                        R_mat, Qf_mat, Q_mat, target_x, u_list, lambda_vector, mu_vector, \
                                        hazard_list, hazard_radius)
        self.u_list = np.array([i for i in result.contents])
        self.u_list = self.u_list.reshape((self.time_horizon, self.u_dim, 1))
        action = self.u_list[0].ravel()
        return action
