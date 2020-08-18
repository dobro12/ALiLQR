from pybullet_utils import bullet_client
import pybullet as p1
import pybullet_data

from scipy.spatial.transform import Rotation as R
from copy import deepcopy
from math import erf
import numpy as np
import random
import time
import copy


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

def diag_mat(diag):
  mat = np.eye(len(diag))
  for i in range(len(diag)):
    mat[i,i] = diag[i]
  return mat


class Cheetah(object):
  def __init__( self, pybullet_client, time_step, useFixedBase=True):
    # set elementary variable
    self.pybullet_client = pybullet_client
    self.time_step = time_step
    self.torque_max = 1000.0
    self.torque_min = -self.torque_max
    self.num_leg = 4

    # init value
    self.init_base_pos = [0.0, 0.0, 0.25]
    self.init_base_orn = self.pybullet_client.getQuaternionFromEuler([0, 0, 0])
    self.init_joint_state = [-0.1, -0.7, 1.5]*self.num_leg
    for joint_idx in range(len(self.init_joint_state)):
      if int(joint_idx/3)%2 != 0 and joint_idx%3 == 0:
        self.init_joint_state[joint_idx] = -self.init_joint_state[joint_idx]

    # load model
    self.sim_model = self.pybullet_client.loadURDF(
        "mini_cheetah/mini_cheetah.urdf", 
        self.init_base_pos,
        self.init_base_orn,
        globalScaling=1.0,
        useFixedBase=useFixedBase,
        flags=self.pybullet_client.URDF_MAINTAIN_LINK_ORDER)

    # joint info
    self.joint_list = []
    self.joint_info_list = []
    for j in range(self.pybullet_client.getNumJoints(self.sim_model)): #12개
      joint_info = self.pybullet_client.getJointInfo(self.sim_model, j)
      if joint_info[2] == 0: #type of joint
        self.joint_list.append(j)
      self.joint_info_list.append(joint_info)

    # motor init
    for j in self.joint_list:
      self.pybullet_client.setJointMotorControl2(self.sim_model, j, self.pybullet_client.VELOCITY_CONTROL, force=0)

    #geometry
    #FR, FL, HR, HL 순서
    self.abduct_org_list =  np.array([[0.19, -0.049, 0.0],  [0.19, 0.049, 0.0], [-0.19, -0.049, 0.0], [-0.19, 0.049, 0.0]])
    self.thigh_org_list =   np.array([[0.0, -0.062, 0.0],   [0.0, 0.062, 0.0],  [0.0, -0.062, 0.0],   [0.0, 0.062, 0.0]])
    self.knee_org_list =    np.array([[0.0, 0.0, -0.209],   [0.0, 0.0, -0.209], [0.0, 0.0, -0.209],   [0.0, 0.0, -0.209]])
    self.foot_org_list =    np.array([[0.0, 0.0, -0.18],    [0.0, 0.0, -0.18],  [0.0, 0.0, -0.18],    [0.0, 0.0, -0.18]])
    #joint axis
    self.abduct_axis_list = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    self.thigh_axis_list = np.array([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0]])
    self.knee_axis_list = np.array([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0]])

    # physics parameter
    self.inertia = [[0.011253, 0, 0], [0, 0.036203, 0], [0, 0, 0.042673]]
    self.leg_mass = 0.54 + 0.634 + 0.064 + 0.15
    self.feet_mass = self.leg_mass*0.1
    self.mass = 1.0*(3.3 + 4*self.leg_mass)
    self.gravity = np.array([0, 0, -9.8])
    self.base_com_pos = np.array([1.0e-3, 0.0, 0.0])

    # for swing leg trajectory
    self.K_ps = np.eye(3) * 1000.0
    self.K_ds = np.eye(3) * 1.0


  def get_torque_from_force(self, force_list):
    base_org, base_orn, base_rot, joint_state, joint_pos_list = self.base_pos, self.base_orn, self.base_rot, self.joint_state, self.joint_pos_list

    torque_list = []
    for leg_idx in range(self.num_leg):
      foot_org = self.foot_org_list[leg_idx]
      knee_org = self.knee_org_list[leg_idx]
      knee_rot = y_rot(-joint_pos_list[2 + leg_idx*3])
      thigh_pos = np.matmul(knee_rot, foot_org) + knee_org

      thigh_org = self.thigh_org_list[leg_idx]
      thigh_rot = y_rot(-joint_pos_list[1 + leg_idx*3])
      abduct_pos = np.matmul(thigh_rot, thigh_pos) + thigh_org

      abduct_org = self.abduct_org_list[leg_idx]
      abduct_rot = x_rot(joint_pos_list[0 + leg_idx*3])
      base_pos = np.matmul(abduct_rot, abduct_pos) + abduct_org

      base_force = np.matmul(np.transpose(base_rot),-force_list[leg_idx])
      abduct_torque = np.dot(np.cross(base_pos - abduct_org, base_force), self.abduct_axis_list[leg_idx])
      abduct_force = np.matmul(np.transpose(abduct_rot), base_force)
      thigh_torque = np.dot(np.cross(abduct_pos - thigh_org, abduct_force), self.thigh_axis_list[leg_idx])
      thigh_force = np.matmul(np.transpose(thigh_rot), abduct_force)
      knee_torque = np.dot(np.cross(thigh_pos - knee_org, thigh_force), self.thigh_axis_list[leg_idx])
      torque_list += [abduct_torque, thigh_torque, knee_torque]
    return torque_list

  def get_Jacobian(self, leg_idx, leg_state):
    l_1 = abs(self.knee_org_list[leg_idx][2])
    l_2 = abs(self.foot_org_list[leg_idx][2])
    a_3 = self.thigh_org_list[leg_idx][1]
    c_1 = np.cos(leg_state[0])
    c_2 = np.cos(leg_state[1])
    c_23 = np.cos(leg_state[1] + leg_state[2])
    s_1 = np.sin(leg_state[0])
    s_2 = np.sin(leg_state[1])
    s_23 = np.sin(leg_state[1] + leg_state[2])
    Jacobian = np.array([[0                                     , l_2*c_23 + l_1*c_2          , l_2*c_23      ],
                        [-a_3*s_1 + l_2*c_1*c_23 + l_1*c_1*c_2  , -l_2*s_1*s_23 - l_1*s_1*s_2 , -l_2*s_1*s_23 ],
                        [a_3*c_1 + l_2*s_1*c_23 + l_1*s_1*c_2   , l_2*c_1*s_23 + l_1*c_1*s_2  , l_2*c_1*s_23  ]])
    return Jacobian

  def get_J_dot(self, leg_idx, leg_state):
    l_1 = abs(self.knee_org_list[leg_idx][2])
    l_2 = abs(self.foot_org_list[leg_idx][2])
    a_3 = self.thigh_org_list[leg_idx][1]
    c_1 = np.cos(leg_state[0])
    c_2 = np.cos(leg_state[1])
    c_23 = np.cos(leg_state[1] + leg_state[2])
    s_1 = np.sin(leg_state[0])
    s_2 = np.sin(leg_state[1])
    s_23 = np.sin(leg_state[1] + leg_state[2])
    J1 = np.array([[0                                     , 0                          , 0             ],
                  [-a_3*c_1 - l_2*s_1*c_23 - l_1*s_1*c_2  , -l_2*c_1*s_23 - l_1*c_1*s_2 , -l_2*c_1*s_23 ],
                  [-a_3*s_1 + l_2*c_1*c_23 + l_1*c_1*c_2  , -l_2*s_1*s_23 - l_1*s_1*s_2 , -l_2*s_1*s_23 ]])

    J2 = np.array([[0                           , -l_2*c_23 - l_1*s_2            , -l_2*s_23  ],
                  [-l_2*c_1*s_23 - l_1*c_1*s_2  , -l_2*s_1*c_23 - l_1*s_1*c_2 , -l_2*s_1*c_23 ],
                  [-l_2*s_1*s_23 - l_1*s_1*s_2  , l_2*c_1*c_23 + l_1*c_1*c_2  , l_2*c_1*c_23  ]])

    J3 = np.array([[0           , -l_2*s_23      , -l_2*s_23      ],
                  [-l_2*c_1*s_23 , -l_2*s_1*c_23 , -l_2*s_1*c_23 ],
                  [-l_2*s_1*s_23 , l_2*c_1*c_23  , l_2*c_1*c_23  ]])
    return J1, J2, J3

  def get_cross_mat(self, vector):
    mat = []
    mat.append([0, -vector[2], vector[1]])
    mat.append([vector[2], 0, -vector[0]])
    mat.append([-vector[1], vector[0], 0])
    return np.array(mat)    

  def get_foot_pos_list(self):
    foot_pos_list = []
    base_foot_pos_list = []
    base_foot_vel_list = []
    for leg_idx in range(self.num_leg):
      foot_org = self.foot_org_list[leg_idx]
      knee_org = self.knee_org_list[leg_idx]
      knee_rot = y_rot(-self.joint_pos_list[2 + leg_idx*3])
      thigh_pos = np.matmul(knee_rot, foot_org) + knee_org
      knee_rot_dot = -y_rot_dot(-self.joint_pos_list[2 + leg_idx*3])*self.joint_vel_list[2 + leg_idx*3]
      thigh_vel = np.matmul(knee_rot_dot, foot_org)

      thigh_org = self.thigh_org_list[leg_idx]
      thigh_rot = y_rot(-self.joint_pos_list[1 + leg_idx*3])
      abduct_pos = np.matmul(thigh_rot, thigh_pos) + thigh_org
      thigh_rot_dot = -y_rot_dot(-self.joint_pos_list[1 + leg_idx*3])*self.joint_vel_list[1 + leg_idx*3]
      abduct_vel = np.matmul(knee_rot_dot, thigh_pos) + np.matmul(thigh_rot, thigh_vel)

      abduct_org = self.abduct_org_list[leg_idx]
      abduct_rot = x_rot(self.joint_pos_list[0 + leg_idx*3])
      base_pos = np.matmul(abduct_rot, abduct_pos) + abduct_org
      base_foot_pos_list.append(base_pos)
      abduct_rot_dot = x_rot_dot(self.joint_pos_list[0 + leg_idx*3])*self.joint_vel_list[0 + leg_idx*3] 
      base_vel = np.matmul(abduct_rot_dot, abduct_pos) + np.matmul(abduct_rot, abduct_vel)
      base_foot_vel_list.append(base_vel)

      abs_pos = np.matmul(self.base_rot, base_pos) + self.base_pos
      foot_pos_list.append(abs_pos)
    return np.array(foot_pos_list), np.array(base_foot_pos_list), np.array(base_foot_vel_list)

  def get_state(self):
    self.base_pos, self.base_orn = self.pybullet_client.getBasePositionAndOrientation(self.sim_model)
    self.base_rot = np.reshape(self.pybullet_client.getMatrixFromQuaternion(self.base_orn), (3,3))
    self.base_rpy = self.pybullet_client.getEulerFromQuaternion(self.base_orn)
    self.base_vel, self.base_ang_vel = self.pybullet_client.getBaseVelocity(self.sim_model)
    self.joint_state = self.pybullet_client.getJointStates(self.sim_model, self.joint_list)
    self.joint_pos_list = [state[0] for state in self.joint_state]
    self.joint_vel_list = [state[1] for state in self.joint_state]
    self.body_contact, self.contact_feet_list = self.check_contact()
    self.foot_pos_list, self.base_foot_pos_list, self.base_foot_vel_list = self.get_foot_pos_list()
    self.com_pos = self.base_pos + np.matmul(self.base_rot, self.base_com_pos)

    state = list(self.base_orn)+list(self. base_vel)+list(self.base_ang_vel)+self.joint_pos_list+self.joint_vel_list+self.contact_feet_list
    return np.array(state)

  def swing_leg_controller(self, leg_idx):
    base_foot_pos = self.base_foot_pos_list[leg_idx]
    base_foot_vel = self.base_foot_vel_list[leg_idx]
    joint_pos = self.joint_pos_list[3*leg_idx:3*(leg_idx+1)]
    joint_vel = self.joint_vel_list[3*leg_idx:3*(leg_idx+1)]

    desired_base_foot_pos = self.abduct_org_list[leg_idx] + self.thigh_org_list[leg_idx] + np.array([0, 0, 0.05-self.base_pos[2]]) + (2*(leg_idx%2)-1)*np.array([0.0, 0.1, 0.0])
    desired_base_foot_vel = np.zeros(3)
    desired_base_foot_acc = np.zeros(3)

    #################
    ## feedforward ##
    ff_torque_list = np.zeros(3)
    mass = self.feet_mass
    g = -self.gravity[1]
    Jacobian = self.get_Jacobian(leg_idx, joint_pos)
    J1_dot, J2_dot, J3_dot = self.get_J_dot(leg_idx, joint_pos)
    J_dot = J1_dot*joint_vel[0] + J2_dot*joint_vel[1] + J3_dot*joint_vel[2]
    round_J = np.zeros((3,3))
    round_J[:,0] = np.matmul(J1_dot, joint_vel)
    round_J[:,1] = np.matmul(J2_dot, joint_vel)
    round_J[:,2] = np.matmul(J3_dot, joint_vel)
    #M(theta)
    ff_torque_list += np.matmul(Jacobian.T, desired_base_foot_acc - np.matmul(J_dot, joint_vel))
    #C(theta, theta_dot)
    C_mat = np.matmul(J_dot.T, Jacobian) + np.matmul(Jacobian.T, J_dot) - np.matmul(round_J.T, Jacobian)
    ff_torque_list += np.matmul(C_mat, joint_vel)
    #G(theta)
    G_mat = np.matmul(self.base_rot.T, -self.gravity)
    G_mat = np.matmul(Jacobian.T, G_mat)
    ff_torque_list += G_mat
    ## feedforward ##
    #################

    differ_base_foot_pos = desired_base_foot_pos - base_foot_pos
    differ_base_foot_vel = desired_base_foot_vel - base_foot_vel
    force_list = np.matmul(self.K_ps, differ_base_foot_pos) + np.matmul(self.K_ds, differ_base_foot_vel)
    swing_torque_list = np.matmul(Jacobian.T, force_list) + mass*ff_torque_list
    return swing_torque_list

  def get_torque_list(self, force_list, contact_list):
    torque_list = self.get_torque_from_force(force_list)
    for swing_leg_idx in range(self.num_leg):
      if contact_list[swing_leg_idx] == 1.0:
        continue
      swing_torque_list = self.swing_leg_controller(swing_leg_idx)
      torque_list[swing_leg_idx*3:(swing_leg_idx + 1)*3] = swing_torque_list
    return torque_list


  def reset_pose(self):
    self.pybullet_client.resetBasePositionAndOrientation(self.sim_model, self.init_base_pos, self.init_base_orn)
    self.pybullet_client.resetBaseVelocity(self.sim_model, np.zeros(3), np.zeros(3))
    for i,j in enumerate(self.joint_list):
      self.pybullet_client.resetJointState(self.sim_model, j, self.init_joint_state[i], 0.0) #velocity = 0.0 으로 setting

  def apply_torque(self, torque_list):
    for i,j in enumerate(self.joint_list):
      torque = np.clip(torque_list[i], self.torque_min, self.torque_max)
      self.pybullet_client.setJointMotorControl2(self.sim_model, j, self.pybullet_client.TORQUE_CONTROL, force=torque)

  def check_contact(self):
    pts = self.pybullet_client.getContactPoints(self.sim_model)
    contact_feet_list = [0 for i in range(4)]
    # 2:FR, 5:FL, 8:RR, 11:RL
    feet_list = [2,6,10,14]
    feet_list2 = [3,7,11,15]
    body_contact = 0 #False
    for p in pts:
      if p[1] == self.sim_model:
        part = p[3]
      elif p[2] == self.sim_model:
        part = p[4]
      if not (part in feet_list or part in feet_list2):
        body_contact = 1 #True
      else:
        contact_feet_list[int((part-2)/4)] = 1
    return body_contact, contact_feet_list

  def get_base_pos(self):
    base_pos, base_orn = self.pybullet_client.getBasePositionAndOrientation(self.sim_model)
    return base_pos

  def reset(self):
    self.reset_pose()
    state = self.get_state()
    return state
