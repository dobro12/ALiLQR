from copy import deepcopy
import numpy as np

def x_rot(t):
  rot = [[1.0,       0.0,        0.0],
          [0.0, np.cos(t), -np.sin(t)],
          [0.0, np.sin(t), np.cos(t)]]
  return np.array(rot)
def y_rot(t):
  rot = [[np.cos(t), 0.0, np.sin(t)],
          [      0.0, 1.0,       0.0],
        [-np.sin(t), 0.0, np.cos(t)]]
  return np.array(rot)
def z_rot(t):
  rot = [[np.cos(t), -np.sin(t), 0.0],
          [np.sin(t),  np.cos(t), 0.0],
          [      0.0,        0.0, 1.0]]
  return np.array(rot)
def rpy_rot(rpy):
    return np.matmul(z_rot(rpy[2]), np.matmul(y_rot(rpy[1]), x_rot(rpy[0])))
def diag_mat(diag):
  mat = np.eye(len(diag))
  for i in range(len(diag)):
    mat[i,i] = diag[i]
  return mat

def x_rot_dot(t):
  rot = [[0.0,       0.0,        0.0],
          [0.0, -np.sin(t), -np.cos(t)],
          [0.0, np.cos(t), -np.sin(t)]]
  return np.array(rot)
def y_rot_dot(t):
  rot = [[-np.sin(t), 0.0, np.cos(t)],
          [      0.0, 0.0,       0.0],
        [-np.cos(t), 0.0, -np.sin(t)]]
  return np.array(rot)
def z_rot_dot(t):
  rot = [[-np.sin(t), -np.cos(t), 0.0],
          [np.cos(t),  -np.sin(t), 0.0],
          [      0.0,        0.0, 0.0]]
  return np.array(rot)


class Agent:
    def __init__(self, env=None):
        self.env = env
        self.x_dim = 3*4
        self.u_dim = 4*3
        self.damping_ratio = 1e-8
        self.learning_rate = 0.5
        self.max_iteration = 10 #10
    
        self.Qf_mat = 100.0*np.eye(self.x_dim)
        self.Qf_mat = np.matmul(self.Qf_mat.T, self.Qf_mat)
        self.Q_mat = 1.0*np.eye(self.x_dim) #diag_mat([1.0]*6 + [0.0]*6)
        self.Q_mat = np.matmul(self.Q_mat.T, self.Q_mat) 
        self.R_mat = 1.0*np.eye(self.u_dim)
        self.R_mat = np.matmul(self.R_mat.T, self.R_mat) 

        #for constraint
        self.const_dim = 17
        self.init_lambda_vector = np.zeros((self.const_dim, 1))
        self.init_mu_vector = np.ones((self.const_dim, 1))
        self.mu_scaling_factor = 2.0
        self.max_lambda = 10.0


    def get_action(self, init_x, init_u_list, delta_time_list, foot_pos_list, contact_phi_list, target_x_list, target_u_list):
        #example :
        #delta_time_list = [0.01, 0.01, 0.01]
        #target_x_list = [1, 1, 1, 1]
        #target_u_list = [0, 0, 0]

        #get x list
        J_value = 0
        x_list = [init_x]
        #u_list = deepcopy(target_u_list)
        u_list = deepcopy(init_u_list)
        lambda_list = [self.init_lambda_vector for i in range(len(delta_time_list) + 1)]
        mu_vector = deepcopy(self.init_mu_vector)
        for t_idx, delta_t in enumerate(delta_time_list):
            x = x_list[t_idx]
            u = u_list[t_idx]
            temp_foot_pos_list = foot_pos_list[t_idx]
            next_x = self.transition(x, u, temp_foot_pos_list, delta_t)
            x_list.append(next_x)
            J_value += 0.5*np.matmul((x_list[t_idx] - target_x_list[t_idx]).T, np.matmul(self.Q_mat, x_list[t_idx] - target_x_list[t_idx])) \
                    + 0.5*np.matmul((u_list[t_idx] - target_u_list[t_idx]).T, np.matmul(self.R_mat, (u_list[t_idx] - target_u_list[t_idx])))
        J_value += 0.5*np.matmul((x_list[len(delta_time_list)] - target_x_list[len(delta_time_list)]).T, np.matmul(self.Qf_mat, x_list[len(delta_time_list)] - target_x_list[len(delta_time_list)]))

        #K, d
        K_list = np.zeros((len(delta_time_list), self.u_dim, self.x_dim))
        d_list = np.zeros((len(delta_time_list), self.u_dim, 1))

        cnt = 0
        while cnt < self.max_iteration:
            pre_J_value = J_value
            cnt += 1

            #update lambda list
            for t_idx in range(len(delta_time_list) + 1):
                temp_foot_pos_list, temp_contact_phi_list = foot_pos_list[t_idx], contact_phi_list[t_idx]
                if t_idx == len(delta_time_list):
                    const_vector = self.get_const(x_list[t_idx], np.zeros((self.u_dim, 1)), temp_foot_pos_list, temp_contact_phi_list)
                else:
                    const_vector = self.get_const(x_list[t_idx], u_list[t_idx], temp_foot_pos_list, temp_contact_phi_list)
                lambda_list[t_idx] = np.clip(lambda_list[t_idx] + mu_vector*const_vector, 0.0, self.max_lambda)
            mu_vector = self.mu_scaling_factor*mu_vector            

            #backward pass
            temp_foot_pos_list, temp_contact_phi_list = foot_pos_list[len(delta_time_list)], contact_phi_list[len(delta_time_list)]
            const_vector = self.get_const(x_list[len(delta_time_list)], np.zeros((self.u_dim, 1)), temp_foot_pos_list, temp_contact_phi_list)
            C_x_mat, C_u_mat = self.get_C_mat(x_list[len(delta_time_list)], np.zeros((self.u_dim, 1)), temp_foot_pos_list, temp_contact_phi_list)
            I_mat = self.get_I_mat(const_vector, lambda_list[len(delta_time_list)], mu_vector)
            P_mat = self.Qf_mat + np.matmul(C_x_mat.T, np.matmul(I_mat, C_x_mat))
            p_vector = np.matmul(self.Qf_mat, x_list[len(delta_time_list)] - target_x_list[len(delta_time_list)]) \
                        + np.matmul(C_x_mat.T, lambda_list[len(delta_time_list)] + np.matmul(I_mat, const_vector))
            for t_idx in range(len(delta_time_list) - 1, -1, -1):
                delta_t = delta_time_list[t_idx]
                temp_foot_pos_list, temp_contact_phi_list = foot_pos_list[t_idx], contact_phi_list[t_idx]
                A_mat, B_mat = self.get_A_B_mat(x_list[t_idx], u_list[t_idx], temp_foot_pos_list, delta_t)
                const_vector = self.get_const(x_list[t_idx], u_list[t_idx], temp_foot_pos_list, temp_contact_phi_list)
                C_x_mat, C_u_mat = self.get_C_mat(x_list[t_idx], u_list[t_idx], temp_foot_pos_list, temp_contact_phi_list)
                I_mat = self.get_I_mat(const_vector, lambda_list[t_idx], mu_vector)

                Qxx = self.Q_mat + np.matmul(A_mat.T, np.matmul(P_mat, A_mat)) + np.matmul(C_x_mat.T, np.matmul(I_mat, C_x_mat))
                Quu = self.R_mat + np.matmul(B_mat.T, np.matmul(P_mat, B_mat)) + np.matmul(C_u_mat.T, np.matmul(I_mat, C_u_mat))
                Qux = np.matmul(B_mat.T, np.matmul(P_mat, A_mat)) + np.matmul(C_u_mat.T, np.matmul(I_mat, C_x_mat))
                Qxu = np.matmul(A_mat.T, np.matmul(P_mat, B_mat)) + np.matmul(C_x_mat.T, np.matmul(I_mat, C_u_mat))
                Qx = np.matmul(self.Q_mat, x_list[t_idx] - target_x_list[t_idx]) + np.matmul(A_mat.T, p_vector) + np.matmul(C_x_mat.T, lambda_list[t_idx] + np.matmul(I_mat, const_vector))
                Qu = np.matmul(self.R_mat, u_list[t_idx] - target_u_list[t_idx]) + np.matmul(B_mat.T, p_vector) + np.matmul(C_u_mat.T, lambda_list[t_idx] + np.matmul(I_mat, const_vector))

                temp_mat = -np.linalg.inv(Quu + self.damping_ratio*np.eye(self.u_dim))
                K_mat = np.matmul(temp_mat, Qux)
                d_vector = np.matmul(temp_mat, Qu)

                P_mat = Qxx + np.matmul(K_mat.T, np.matmul(Quu, K_mat)) + np.matmul(K_mat.T, Qux) + np.matmul(Qxu, K_mat)
                p_vector = Qx + np.matmul(K_mat.T, np.matmul(Quu, d_vector)) + np.matmul(K_mat.T, Qu) + np.matmul(Qxu, d_vector)
                K_list[t_idx] = K_mat
                d_list[t_idx] = d_vector

            '''
            #forward pass
            new_x_list = [init_x]
            new_u_list = []
            J_value = 0
            for t_idx, delta_t in enumerate(delta_time_list)):
                delta_x = new_x_list[t_idx] - x_list[t_idx]
                delta_u = np.matmul(K_list[t_idx], delta_x) + self.learning_rate*d_list[t_idx]
                new_u_list.append(u_list[t_idx] + delta_u)
                temp_foot_pos_list = foot_pos_list[t_idx]
                next_x = self.transition(new_x_list[t_idx], new_u_list[t_idx], temp_foot_pos_list, delta_t)
                new_x_list.append(next_x)
                J_value += 0.5*np.matmul((new_x_list[t_idx] - target_x_list[t_idx]).T, np.matmul(self.Q_mat, new_x_list[t_idx] - target_x_list[t_idx])) \
                        + 0.5*np.matmul((new_u_list[t_idx] - target_u_list[t_idx]).T, np.matmul(self.R_mat, (new_u_list[t_idx] - target_u_list[t_idx])))
            J_value += 0.5*np.matmul((new_x_list[len(delta_time_list)] - target_x_list[len(delta_time_list)]).T, np.matmul(self.Qf_mat, new_x_list[len(delta_time_list)] - target_x_list[len(delta_time_list)]))

            print(pre_J_value - J_value, J_value)
            #print(new_x_list[-1][:3,0])
            if pre_J_value - J_value < 0.0:
                break
            '''
            #forward pass
            learning_rate = self.learning_rate
            while True:
                new_x_list = [init_x]
                new_u_list = []
                J_value = 0
                for t_idx, delta_t in enumerate(delta_time_list):
                    delta_x = new_x_list[t_idx] - x_list[t_idx]
                    delta_u = np.matmul(K_list[t_idx], delta_x) + learning_rate*d_list[t_idx]
                    new_u_list.append(u_list[t_idx] + delta_u)
                    temp_foot_pos_list = foot_pos_list[t_idx]
                    next_x = self.transition(new_x_list[t_idx], new_u_list[t_idx], temp_foot_pos_list, delta_t)
                    new_x_list.append(next_x)
                    J_value += 0.5*np.matmul((new_x_list[t_idx] - target_x_list[t_idx]).T, np.matmul(self.Q_mat, new_x_list[t_idx] - target_x_list[t_idx])) \
                            + 0.5*np.matmul((new_u_list[t_idx] - target_u_list[t_idx]).T, np.matmul(self.R_mat, (new_u_list[t_idx] - target_u_list[t_idx])))
                J_value += 0.5*np.matmul((new_x_list[len(delta_time_list)] - target_x_list[len(delta_time_list)]).T, np.matmul(self.Qf_mat, new_x_list[len(delta_time_list)] - target_x_list[len(delta_time_list)]))
                if pre_J_value - J_value >= 0.0:
                    break
                learning_rate *= 0.5
            print(pre_J_value - J_value, J_value)
            #print(new_x_list[-1][:3,0])
            x_list = new_x_list
            u_list = new_u_list

        delta_x = new_x_list[0] - x_list[0]
        delta_u = np.matmul(K_list[0], delta_x) + d_list[0]
        return u_list[0] + delta_u, u_list

    def transition(self, x, u, foot_pos_list, delta_t):
        assert len(x) == self.x_dim and len(u) == self.u_dim

        com_pos = x[:3, 0]
        rpy = x[3:6, 0]
        com_vel = x[6:9, 0]
        base_ang_vel = x[9:, 0]

        force_list = u.reshape((4,3))
        f_net = np.sum(force_list, axis=0)
        base_torque_net = np.sum([np.cross(foot_pos_list[i] - com_pos, force_list[i]) for i in range(4)], axis=0)
        base_torque_net = np.matmul(z_rot(rpy[2]).T, base_torque_net)
        
        next_com_pos = com_pos + delta_t*com_vel + 0.5*(delta_t**2)*(f_net/self.env.model.mass + self.env.model.gravity)
        next_rpy = rpy + delta_t*base_ang_vel + 0.5*(delta_t**2)*np.matmul(np.linalg.inv(self.env.model.inertia), base_torque_net)
        next_com_vel = com_vel + delta_t*(f_net/self.env.model.mass + self.env.model.gravity)
        next_base_ang_vel = base_ang_vel + delta_t*np.matmul(np.linalg.inv(self.env.model.inertia), base_torque_net)

        new_x = np.concatenate([next_com_pos.ravel(), next_rpy.ravel(), next_com_vel.ravel(), next_base_ang_vel.ravel()])
        new_x = new_x.reshape((-1, 1))
        return new_x

    def get_const(self, x, u, foot_pos_list, contact_phi_list):
        com_pos = x[:3, 0]
        rpy = x[3:6, 0]
        com_vel = x[6:9, 0]
        base_ang_vel = x[9:, 0]
        force_list = u.reshape((4,3))

        z_const = 0.4 * 0.9
        ground_normal = np.array([0, 0, 1])
        friction_coeff = 0.9

        const = np.zeros((self.const_dim, 1))
        const_idx = 0
        get_dist = lambda x : np.sqrt(np.dot(x, x))

        # kinematic leg limits
        base_rot = rpy_rot(rpy)
        for leg_idx in range(4):
            heap_pos = np.matmul(base_rot, self.env.model.abduct_org_list[leg_idx]) + com_pos
            leg_dist = get_dist((foot_pos_list[leg_idx] - heap_pos).ravel())
            const[const_idx, 0] = (leg_dist - z_const)*contact_phi_list[leg_idx]
            const_idx += 1

        # swing leg's force limits
        temp_const = 0
        for leg_idx in range(4):
            temp_const += (1 - contact_phi_list[leg_idx])*get_dist(force_list[leg_idx])
        const[const_idx, 0] = temp_const
        const_idx += 1

        # positive ground force normal
        for leg_idx in range(4):
            const[const_idx, 0] = -np.dot(force_list[leg_idx], ground_normal)
            const_idx += 1

        # friction pyramid
        for leg_idx in range(4):
            const[const_idx, 0] = np.abs(force_list[leg_idx, 0]) - friction_coeff*force_list[leg_idx, 2]
            const_idx += 1
            const[const_idx, 0] = np.abs(force_list[leg_idx, 1]) - friction_coeff*force_list[leg_idx, 2]
            const_idx += 1

        return const

    def get_A_B_mat(self, x, u, foot_pos_list, delta_t):
        new_x = self.transition(x, u, foot_pos_list, delta_t)
        EPS = 1e-3

        A = []
        for x_idx in range(self.x_dim):
            delta_x = np.zeros_like(x)
            delta_x[x_idx][0] = EPS
            A.append((self.transition(x + delta_x, u, foot_pos_list, delta_t) - new_x)/EPS)
        A = np.concatenate(A, axis=-1)

        B = []
        for u_idx in range(self.u_dim):
            delta_u = np.zeros_like(u)
            delta_u[u_idx][0] = EPS
            B.append((self.transition(x, u + delta_u, foot_pos_list, delta_t) - new_x)/EPS)
        B = np.concatenate(B, axis=-1)
        return A, B

    def get_C_mat(self, x, u, foot_pos_list, contact_phi_list):
        const = self.get_const(x, u, foot_pos_list, contact_phi_list)
        EPS = 1e-3

        C_x_mat = []
        for x_idx in range(self.x_dim):
            delta_x = np.zeros_like(x)
            delta_x[x_idx][0] = EPS
            C_x_mat.append((self.get_const(x + delta_x, u, foot_pos_list, contact_phi_list) - const)/EPS)
        C_x_mat = np.concatenate(C_x_mat, axis=-1)

        if len(u) == 0:
            C_u_mat = []
        else:
            C_u_mat = []
            for u_idx in range(self.u_dim):
                delta_u = np.zeros_like(u)
                delta_u[u_idx][0] = EPS
                C_u_mat.append((self.get_const(x, u + delta_u, foot_pos_list, contact_phi_list) - const)/EPS)
            C_u_mat = np.concatenate(C_u_mat, axis=-1)
        return C_x_mat, C_u_mat

    def get_I_mat(self, const_vector, lambda_vector, mu_vector):
        I_mat = np.eye(self.const_dim)
        for const_idx in range(self.const_dim):
            if const_vector[const_idx] >= 0:
                I_mat[const_idx, const_idx] = mu_vector[const_idx]
        return I_mat
