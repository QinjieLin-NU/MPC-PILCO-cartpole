import logging
import math
from math import *
import gym
import numpy as np

from gym import make as gym_make
from gym import spaces
from gym.utils import seeding

logger = logging.getLogger(__name__)


class CartPoleSwingUpContinuousEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.g = 9.82  # gravity
        self.m_c = 0.5  # cart mass
        self.m_p = 0.5  # pendulum mass
        self.total_m = (self.m_p + self.m_c)
        self.l = 0.6  # pole's length
        self.m_p_l = (self.m_p * self.l)
        self.force_mag = 10.0
        self.dt = 0.01  # seconds between state updates
        self.b = 0.1  # friction coefficient

        self.t = 0  # timestep
        self.t_limit = 500

        # Angle, angle speed and speed at which to fail the episode
        self.x_threshold = 6
        self.x_dot_threshold = 10
        self.theta_dot_threshold = 10

        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(-10.0, 10.0, shape=(1,))
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # valid action
        action = np.clip(action, -10.0, 10.0)[0]

        state = self.state
        x, x_dot, theta, theta_dot = state

        # nullify action, if it would normally push the cart out of boundaries
        if x >= self.x_threshold and action >= 10:
            action = 0
        elif x <= -self.x_threshold and action <= -10:
            action = 0

        s = math.sin(theta)
        c = math.cos(theta)

        #mass of cart and pole
        mp = self.m_p
        mc =  self.m_c
        l = self.m_p_l
        U = action
        g=self.g
        dx = x_dot
        q = theta
        dq = theta_dot

        ddx = (U + mp * sin(q) * (l * dq * dq + g * cos(q))) / (
                mc + mp * sin(q) * sin(q))  # acceleration of x
        ddq = (-U * cos(q) - mp * l * dq * dq * sin(q) * cos(q) - (
                mc + mp) * g * sin(
            q)) / (
                        l * mc + l * mp * sin(q) * sin(q))  # acceleration of theta
        xdot_update = ddx
        thetadot_update = ddq
        # xdot_update = (-2 * self.m_p_l * (
        #         theta_dot ** 2) * s + 3 * self.m_p * self.g * s * c + 4 * action - 4 * self.b * x_dot) / (
        #                       4 * self.total_m - 3 * self.m_p * c ** 2)
        # thetadot_update = (-3 * self.m_p_l * (theta_dot ** 2) * s * c + 6 * self.total_m * self.g * s + 6 * (
        #         action - self.b * x_dot) * c) / (4 * self.l * self.total_m - 3 * self.m_p_l * c ** 2)

        x = x + x_dot * self.dt
        theta = theta + theta_dot * self.dt
        x_dot = x_dot + xdot_update * self.dt
        theta_dot = theta_dot + thetadot_update * self.dt

        self.state = (x, x_dot, theta, theta_dot)

        done = False

        # restrict state of cart to be within its limits without terminating the game
        if x > self.x_threshold:
            x = self.x_threshold
        elif x < -self.x_threshold:
            x = -self.x_threshold
        elif x_dot > self.x_dot_threshold:
            x_dot = self.x_dot_threshold
        elif x_dot < -self.x_dot_threshold:
            x_dot = -self.x_dot_threshold
        elif theta_dot > self.theta_dot_threshold:
            theta_dot = self.theta_dot_threshold
        elif theta_dot < -self.theta_dot_threshold:
            theta_dot = -self.theta_dot_threshold

        self.t += 1

        # terminate the game if t >= time limit
        if self.t >= self.t_limit:
            done = True

        # reward function as described in dissertation of Deisenroth with A=1
        A = 1
        invT = A * np.array([[1, self.l, 0], [self.l, self.l ** 2, 0], [0, 0, self.l ** 2]])
        j = np.array([x, np.sin(theta), np.cos(theta)])
        j_target = np.array([0.0, 0.0, 1.0])

        reward = np.matmul((j - j_target), invT)
        reward = np.matmul(reward, (j - j_target))
        reward = -(1 - np.exp(-0.5 * reward))

        obs = np.array([x, x_dot, theta, theta_dot])

        return obs, reward, done, {}

    def reset(self):
        # spawn cart at the same initial state + randomness
        self.state = np.random.normal(loc=np.array([0.0, 0.0, np.pi, 0.0]), scale=np.array([0.01, 0.01, 0.01, 0.01]))
        self.t = 0  # timestep
        obs = self.state
        return obs

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 600

        world_width = 5
        scale = screen_width / world_width
        carty = screen_height / 2
        polewidth = 6.0
        polelen = scale * self.l
        cartwidth = 40.0
        cartheight = 20.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2

            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            cart.set_color(1, 0, 0)
            self.viewer.add_geom(cart)

            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0, 0, 1)
            self.poletrans = rendering.Transform(translation=(0, 0))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)

            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.1, 1, 1)
            self.viewer.add_geom(self.axle)

            self.pole_bob = rendering.make_circle(polewidth / 2)
            self.pole_bob_trans = rendering.Transform()
            self.pole_bob.add_attr(self.pole_bob_trans)
            self.pole_bob.add_attr(self.poletrans)
            self.pole_bob.add_attr(self.carttrans)
            self.pole_bob.set_color(0, 0, 0)
            self.viewer.add_geom(self.pole_bob)

            self.wheel_l = rendering.make_circle(cartheight / 4)
            self.wheel_r = rendering.make_circle(cartheight / 4)
            self.wheeltrans_l = rendering.Transform(translation=(-cartwidth / 2, -cartheight / 2))
            self.wheeltrans_r = rendering.Transform(translation=(cartwidth / 2, -cartheight / 2))
            self.wheel_l.add_attr(self.wheeltrans_l)
            self.wheel_l.add_attr(self.carttrans)
            self.wheel_r.add_attr(self.wheeltrans_r)
            self.wheel_r.add_attr(self.carttrans)
            self.wheel_l.set_color(0, 0, 0)
            self.wheel_r.set_color(0, 0, 0)
            self.viewer.add_geom(self.wheel_l)
            self.viewer.add_geom(self.wheel_r)

            self.track = rendering.Line(
                (screen_width / 2 - self.x_threshold * scale, carty - cartheight / 2 - cartheight / 4),
                (screen_width / 2 + self.x_threshold * scale, carty - cartheight / 2 - cartheight / 4))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(x[2])
        self.pole_bob_trans.set_translation(-self.l * np.sin(x[2]), self.l * np.cos(x[2]))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
