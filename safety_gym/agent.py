from car_env.env import Env
import numpy as np

class Agent:
    def __init__(self, env=None):
        self.x_dim = env.observation_space.shape[0]
        self.u_dim = env.action_space.shape[0]
        self.delta_t = env.time_step
        self.time_horizon = 20
        self.damping_ratio = 1e-4
        self.learning_rate = 1e-2
    
        self.R_mat = np.array([[0.1, 0.0], [0.0, 0.1]])
        self.R_mat = np.matmul(self.R_mat.T, self.R_mat) 
        self.Qf_mat = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 0.0]])
        self.Qf_mat = np.matmul(self.Qf_mat.T, self.Qf_mat)
        self.Q_mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        self.Q_mat = np.matmul(self.Q_mat.T, self.Q_mat) 
        self.target_x = np.array([[0.0], [0.0], [0.0]])

        #init u list
        self.u_list = np.zeros((self.time_horizon, self.u_dim, 1)) #u : 1x1, u_list : N*1*1


    def reset(self):
        #init u list
        self.u_list = np.zeros((self.time_horizon, self.u_dim, 1)) #u : 1x1, u_list : N*1*1

    def get_action(self, init_x):
        time_horizon = self.time_horizon
        delta_t = self.delta_t
        damping_ratio = self.damping_ratio
        learning_rate = self.learning_rate
        u_dim = self.u_dim
        x_dim = self.x_dim

        R_mat = self.R_mat
        Qf_mat = self.Qf_mat
        Q_mat = self.Q_mat
        target_x = self.target_x
        u_list = self.u_list

        #get x list
        J_value = 0
        x_list = [init_x]
        for t_idx in range(time_horizon):
            x = x_list[t_idx]
            u = u_list[t_idx]
            next_x = self.transition(x, u)
            x_list.append(next_x)
            J_value += 0.5*np.matmul((x_list[t_idx] - target_x).T, np.matmul(Q_mat, x_list[t_idx] - target_x)) \
                    + 0.5*np.matmul(u_list[t_idx].T, np.matmul(R_mat, u_list[t_idx]))
        J_value += 0.5*np.matmul((x_list[time_horizon] - target_x).T, np.matmul(Qf_mat, x_list[time_horizon] - target_x))

        #K, d
        K_list = np.zeros((time_horizon, u_dim, x_dim))
        d_list = np.zeros((time_horizon, u_dim, 1))

        max_cnt = 10
        cnt = 0
        while max_cnt > cnt:
            pre_J_value = J_value
            cnt += 1

            #backward pass
            P_mat = Qf_mat
            p_vector = np.matmul(Qf_mat, x_list[time_horizon] - target_x)
            for t_idx in range(time_horizon - 1, -1, -1):
                A_mat, B_mat = self.get_A_B(x_list[t_idx], u_list[t_idx])
                Qxx = Q_mat + np.matmul(A_mat.T, np.matmul(P_mat, A_mat))
                Quu = R_mat + np.matmul(B_mat.T, np.matmul(P_mat, B_mat))
                Qux = np.matmul(B_mat.T, np.matmul(P_mat, A_mat))
                Qxu = np.matmul(A_mat.T, np.matmul(P_mat, B_mat))
                Qx = np.matmul(Q_mat, x_list[t_idx] - target_x) + np.matmul(A_mat.T, p_vector)
                Qu = np.matmul(R_mat, u_list[t_idx]) + np.matmul(B_mat.T, p_vector)

                temp_mat = -np.linalg.inv(Quu + damping_ratio*np.eye(u_dim))
                K_mat = np.matmul(temp_mat, Qux)
                d_vector = np.matmul(temp_mat, Qu)

                P_mat = Qxx + np.matmul(K_mat.T, np.matmul(Quu, K_mat)) + np.matmul(K_mat.T, Qux) + np.matmul(Qxu, K_mat)
                p_vector = Qx + np.matmul(K_mat.T, np.matmul(Quu, d_vector)) + np.matmul(K_mat.T, Qu) + np.matmul(Qxu, d_vector)
                K_list[t_idx] = K_mat
                d_list[t_idx] = d_vector

            #forward pass
            new_x_list = [init_x]
            new_u_list = []
            J_value = 0
            for t_idx in range(time_horizon):
                delta_x = new_x_list[t_idx] - x_list[t_idx]
                delta_u = np.matmul(K_list[t_idx], delta_x) + learning_rate*d_list[t_idx]
                new_u_list.append(u_list[t_idx] + delta_u)
                next_x = self.transition(new_x_list[t_idx], new_u_list[t_idx])
                new_x_list.append(next_x)
                J_value += 0.5*np.matmul((new_x_list[t_idx] - target_x).T, np.matmul(Q_mat, new_x_list[t_idx] - target_x)) \
                        + 0.5*np.matmul(new_u_list[t_idx].T, np.matmul(R_mat, new_u_list[t_idx]))
            J_value += 0.5*np.matmul((new_x_list[time_horizon] - target_x).T, np.matmul(Qf_mat, new_x_list[time_horizon] - target_x))

            x_list = new_x_list
            u_list = new_u_list
            #print(pre_J_value - J_value, J_value)
        self.u_list = u_list
        return u_list[0]

    def transition(self, x, u):
        assert len(x) == self.x_dim and len(u) == self.u_dim
        delta_t = self.delta_t

        b = 0.13
        a = -0.1
        r = 0.05

        u1, u2 = u[0][0], u[1][0]
        x, y, theta = x[0][0], x[1][0], x[2][0]

        omega = r*(u2 - u1)/(2*b)
        vel_x = r*u1*np.cos(theta) + omega*(a*np.sin(theta) + b*np.cos(theta))
        vel_y = r*u1*np.sin(theta) + omega*(b*np.sin(theta) - a*np.cos(theta))
        x += vel_x*delta_t
        y += vel_y*delta_t
        theta += delta_t*omega

        new_x = np.array([[x], [y], [theta]])
        return new_x

    def get_A_B(self, x, u):
        new_x = self.transition(x, u)
        EPS = 1e-3

        A = []
        for x_idx in range(self.x_dim):
            delta_x = np.zeros_like(x)
            delta_x[x_idx][0] = EPS
            A.append((self.transition(x + delta_x, u) - new_x)/EPS)
        A = np.concatenate(A, axis=-1)

        B = []
        for u_idx in range(self.u_dim):
            delta_u = np.zeros_like(u)
            delta_u[u_idx][0] = EPS
            B.append((self.transition(x, u + delta_u) - new_x)/EPS)
        B = np.concatenate(B, axis=-1)
        return A, B

    def forward_dynamics(self, init_x, u_list):
        x_list = [init_x]
        for t_idx in range(self.time_horizon):
            x = x_list[t_idx]
            u = u_list[t_idx]
            next_x = self.transition(x, u)
            x_list.append(next_x)
        return x_list

if __name__ == "__main__":
    env = Env()
    agent = Agent(env)