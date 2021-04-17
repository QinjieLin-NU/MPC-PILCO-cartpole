import gym
import torch
import math
import numpy as np
import torch
from math import *


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

class DifferentiableCartPole():
    def __init__(self):
        self.dt = 0.04
        self.g = 10.0
        self.force_mag = 20.0

        def _sysID_dynamics(physics_params, state, action):
            state = state.flatten()
            x, x_dot, theta, theta_dot = state[0::4],state[1::4],state[2::4],state[3::4]
            x, x_dot, theta, theta_dot = x.unsqueeze(1), x_dot.unsqueeze(1), theta.unsqueeze(1), theta_dot.unsqueeze(1)
            force = action.flatten().unsqueeze(1) * self.force_mag

            #calculate xacc & thetaacc using PDP
            dx = x_dot
            q = theta
            dq = theta_dot
            U = force

            #mass of cart and pole
            mp,mc,l = physics_params
            g=self.g

            m_c = mc  # cart mass, default 0.5
            m_p = mp  # pendulum mass, default 0.5
            total_m = (m_p + m_c)
            m_p_l = (m_p * l)
            s = torch.sin(theta)
            c = torch.cos(theta)
            self.b = 0.1 #friction
            xacc = (-2 * m_p_l * (
                        theta_dot ** 2) * s + 3 * m_p * g * s * c + 4 * action - 4 * self.b * x_dot) / (
                                    4 * total_m - 3 * m_p * c ** 2)
            thetaacc = (-3 * m_p_l * (theta_dot ** 2) * s * c + 6 * total_m * g * s + 6 * (
                        action - self.b * x_dot) * c) / (4 * l * total_m - 3 * m_p_l * c ** 2)

            # ddx = (U + mp * torch.sin(q) * (l * dq * dq + g * torch.cos(q))) / (
            #         mc + mp * torch.sin(q) * torch.sin(q))  # acceleration of x
            # ddq = (-U * torch.cos(q) - mp * l * dq * dq * torch.sin(q) * torch.cos(q) - (
            #         mc + mp) * g * torch.sin(q)) / (l * mc + l * mp * torch.sin(q) * torch.sin(q))  # acceleration of theta
            # xacc = ddx
            # thetaacc = ddq

            delta_x = self.dt * x_dot
            delta_xDot = self.dt * xacc
            delta_theta = self.dt * theta_dot
            delta_thetaDot = self.dt * thetaacc
            delta_state = torch.cat([delta_x, delta_xDot, delta_theta, delta_thetaDot],dim=1)
            return delta_state

        self.sysID_dynamics = _sysID_dynamics
