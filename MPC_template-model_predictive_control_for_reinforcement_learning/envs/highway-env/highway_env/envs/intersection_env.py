from __future__ import division, print_function, absolute_import
from gym.envs.registration import register
from gym import spaces
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane, AbstractLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.control import MDPVehicle, ControlledLowLevelVehicle
from highway_env.vehicle.dynamics import Vehicle


class IntersectionEnv(AbstractEnv):
    COLLISION_REWARD = -5
    HIGH_VELOCITY_REWARD = 1
    ARRIVED_REWARD = 1

    ACTIONS = {
        0: 'SLOWER',
        1: 'IDLE',
        2: 'FASTER'
    }
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "absolute": True,
                "flatten": False,
                "observe_intentions": False
            },
            "duration": 13,  # [s]
            "destination": "o1",
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5 * 1.3,
            "collision_reward": IntersectionEnv.COLLISION_REWARD,
            "normalize_reward": False,
            "range": None,
            "num_vehicles": 2
        })
        return config

    def _reward(self, action):
        reward = self.config["collision_reward"] * self.vehicle.crashed \
                 + self.HIGH_VELOCITY_REWARD * (self.vehicle.velocity_index == self.vehicle.SPEED_COUNT - 1)
        reward = self.ARRIVED_REWARD if self.has_arrived else reward
        if self.config["normalize_reward"]:
            reward = utils.remap(reward, [self.config["collision_reward"], self.ARRIVED_REWARD], [0, 1])
        return reward

    def _is_terminal(self):
        """
            The episode is over when a collision occurs or when the access ramp has been passed.
        """
        return self.vehicle.crashed \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"] \
               or self.has_arrived

    def reset(self):
        self._make_road()
        self._make_vehicles(self.config["num_vehicles"])
        self.steps = 0
        return super(IntersectionEnv, self).reset()

    def step(self, action):
        results = super(IntersectionEnv, self).step(action)
        self.steps += 1
        self._clear_vehicles()
        self._spawn_vehicle()
        return results

    def _make_road(self):
        """
            Make an 4-way intersection.

            The horizontal road has the right of way. More precisely, the levels of priority are:
                - 3 for horizontal straight lanes and right-turns
                - 1 for vertical straight lanes and right-turns
                - 2 for horizontal left-turns
                - 0 for vertical left-turns
            The code for nodes in the road network is:
            (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)
        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # Incoming
            start = rotation @ np.array([lane_width / 2, access_length + outer_distance])
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane("o" + str(corner), "ir" + str(corner),
                         StraightLane(start, end, line_types=[s, c], priority=priority, speed_limit=10))
            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane("ir" + str(corner), "il" + str((corner - 1) % 4),
                         CircularLane(r_center, right_turn_radius, angle + np.radians(180), angle + np.radians(270),
                                      line_types=[n, c], priority=priority, speed_limit=10))
            # Left turn
            l_center = rotation @ (np.array([-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2]))
            net.add_lane("ir" + str(corner), "il" + str((corner + 1) % 4),
                         CircularLane(l_center, left_turn_radius, angle + np.radians(0), angle + np.radians(-90),
                                      clockwise=False, line_types=[n, n], priority=priority - 1, speed_limit=10))
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane("ir" + str(corner), "il" + str((corner + 2) % 4),
                         StraightLane(start, end, line_types=[s, n], priority=priority, speed_limit=10))
            # Exit
            start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane("il" + str((corner - 1) % 4), "o" + str((corner - 1) % 4),
                         StraightLane(end, start, line_types=[n, c], priority=priority, speed_limit=10))

        road = RegulatedRoad(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self, n_vehicles=10):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 2  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [(self.road.act(), self.road.step(1 / self.SIMULATION_FREQUENCY)) for _ in range(self.SIMULATION_FREQUENCY)]

        # Challenger vehicle
        self._spawn_vehicle(60, spawn_probability=1, go_straight=True, position_deviation=0.1, velocity_deviation=0)

        # Ego-vehicle
        MDPVehicle.SPEED_MIN = 0
        MDPVehicle.SPEED_MAX = 9
        MDPVehicle.SPEED_COUNT = 3
        # MDPVehicle.TAU_A = 1.0
        ego_lane = self.road.network.get_lane(("o0", "ir0", 0))
        destination = self.config["destination"] or "o" + str(self.np_random.randint(1, 4))
        ego_vehicle = MDPVehicle(self.road,
                                 ego_lane.position(60, 0),
                                 velocity=ego_lane.speed_limit,
                                 heading=ego_lane.heading_at(50)) \
            .plan_route_to(destination)
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        for v in self.road.vehicles:  # Prevent early collisions
            if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 20:
                self.road.vehicles.remove(v)

    def _spawn_vehicle(self,
                       longitudinal=0,
                       position_deviation=1.,
                       velocity_deviation=1.,
                       spawn_probability=0.6,
                       go_straight=False):
        if self.np_random.rand() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(self.road, ("o" + str(route[0]), "ir" + str(route[0]), 0),
                                            longitudinal=longitudinal + 5 + self.np_random.randn() * position_deviation,
                                            velocity=8 + self.np_random.randn() * velocity_deviation)
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self):
        is_leaving = lambda vehicle: "il" in vehicle.lane_index[0] and "o" in vehicle.lane_index[1] \
                                     and vehicle.lane.local_coordinates(vehicle.position)[0] \
                                     >= vehicle.lane.length - 4 * vehicle.LENGTH
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle is self.vehicle or not (is_leaving(vehicle) or vehicle.route is None)]

    @property
    def has_arrived(self):
        return "il" in self.vehicle.lane_index[0] \
               and "o" in self.vehicle.lane_index[1] \
               and self.vehicle.lane.local_coordinates(self.vehicle.position)[0] >= \
               self.vehicle.lane.length - 3 * self.vehicle.LENGTH

    def _cost(self, action):
        """
            The constraint signal is the occurence of collisions.
        """
        return float(self.vehicle.crashed)


class IntersectionLowLevelControlEnv(AbstractEnv):
    COLLISION_REWARD = -5
    HIGH_VELOCITY_REWARD = 1
    POS_ARRIVED_REWARD = 2
    HEADING_ARRIVED_REWARD = 1
    STEERING_RANGE = np.pi / 4
    ACCELERATION_RANGE = 5.0

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 2,
                "features": ["presence", "id", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-40, 40],
                    "y": [-40, 40],
                    "vx": [-70, 70],
                    "vy": [-70, 70],
                },
                "absolute": False,
                "order": "notsorted",
                "flatten": False,
                "observe_intentions": False
            },
            "duration": 50,  # [s]
            "policy_frequency": 10,  # [Hz]
            "destination": "o1",
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5 * 1.3,
            "collision_reward": IntersectionLowLevelControlEnv.COLLISION_REWARD,
            "normalize_reward": False,
            "range": 40,
            "num_vehicles": 0,
            "spawn_frequency": 10,
            "spawn": False,
            "route": [1, 2]
        })
        return config

    def define_spaces(self):
        super().define_spaces()
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype=np.float32)

    def _reward(self, action):
        heading = np.arctan2(np.sin(self.vehicle.heading), np.cos(self.vehicle.heading))
        range = self.config["observation"]["features_range"]["x"][1]

        r_crash = self.config["collision_reward"] * self.vehicle.crashed
        reward = r_crash
        r_vel = self.HIGH_VELOCITY_REWARD * (self.vehicle.velocity/self.vehicle.MAX_VELOCITY)
        x_distance = np.sqrt((self.goal_pos[0] - self.vehicle.position[0]) ** 2)/range
        y_distance = np.sqrt((self.goal_pos[1] - self.vehicle.position[1]) ** 2)/range
        heading_distance = np.abs(self.goal_heading - np.cos(self.vehicle.heading))[0]
        r_x = self.POS_ARRIVED_REWARD * np.exp(- x_distance)
        r_y = self.POS_ARRIVED_REWARD * np.exp(- y_distance)
        r_heading = self.HEADING_ARRIVED_REWARD * np.exp(-heading_distance)

        # reward += r_x
        reward += r_y
        reward += r_heading
        reward += r_vel

        # print('-----------------------------------------------------')
        # print('goal pos: ', self.goal_pos, ' ego pos: ', self.vehicle.position)
        # print('goal heading: ', self.goal_heading, ' ego heading: ', heading)
        # print('r_crash: ', r_crash,' r_y: ', r_y, ' r_heading: ', r_heading, ' r_vel: ', r_vel)

        return reward

    def _is_terminal(self):
        """
            The episode is over when a collision occurs or when the access ramp has been passed.
        """
        # print('steps: ', self.steps)
        # print('total length', self.config["duration"] * self.config["policy_frequency"])
        return self.vehicle.crashed \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"] \
               or self.has_arrived

    def reset(self):
        self._make_road()
        self._make_vehicles(self.config["num_vehicles"])
        self.steps = 0
        self.goal_pos = np.array([-10, -2])
        self.goal_heading = np.array([-1])
        return super(IntersectionLowLevelControlEnv, self).reset()

    def step(self, action):
        self.vehicle.act({
            "acceleration": action[0] * self.ACCELERATION_RANGE,
            "steering": action[1] * self.STEERING_RANGE # Initial 0.55 times
        })
        self._simulate()

        obs = self.observation.observe()
        reward = self._reward(action)
        terminal = self._is_terminal()

        info = {
            "is_success": self.has_arrived,
            "velocity": self.vehicle.velocity,
            "crashed": self.vehicle.crashed,
            "action": action,
        }

        self.steps += 1
        self._clear_vehicles()
        # TODO: the vehicles may crash by themselves
        if self.config["spawn"] and self.steps % self.config["spawn_frequency"] == 0:
            self._spawn_vehicle()
        return obs, reward, terminal, info

    def _make_road(self):
        """
            Make an 4-way intersection.

            The horizontal road has the right of way. More precisely, the levels of priority are:
                - 3 for horizontal straight lanes and right-turns
                - 1 for vertical straight lanes and right-turns
                - 2 for horizontal left-turns
                - 0 for vertical left-turns
            The code for nodes in the road network is:
            (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)
        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # Incoming
            start = rotation @ np.array([lane_width / 2, access_length + outer_distance])
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane("o" + str(corner), "ir" + str(corner),
                         StraightLane(start, end, line_types=[s, c], priority=priority, speed_limit=10))
            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane("ir" + str(corner), "il" + str((corner - 1) % 4),
                         CircularLane(r_center, right_turn_radius, angle + np.radians(180), angle + np.radians(270),
                                      line_types=[n, c], priority=priority, speed_limit=10))
            # Left turn
            l_center = rotation @ (np.array([-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2]))
            net.add_lane("ir" + str(corner), "il" + str((corner + 1) % 4),
                         CircularLane(l_center, left_turn_radius, angle + np.radians(0), angle + np.radians(-90),
                                      clockwise=False, line_types=[n, n], priority=priority - 1, speed_limit=10))
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane("ir" + str(corner), "il" + str((corner + 2) % 4),
                         StraightLane(start, end, line_types=[s, n], priority=priority, speed_limit=10))
            # Exit
            start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane("il" + str((corner - 1) % 4), "o" + str((corner - 1) % 4),
                         StraightLane(end, start, line_types=[n, c], priority=priority, speed_limit=10))

        # TODO: double check here change from regulated to general road

        road = RegulatedRoad(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self, n_vehicles=10):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 2  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3

        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [(self.road.act(), self.road.step(1 / self.SIMULATION_FREQUENCY)) for _ in range(self.SIMULATION_FREQUENCY)]

        # Challenger vehicle
        route = self.config['route']
        if self.config['route'] is None:
            route_to = self.np_random.choice([0,2,3], size=1, replace=False)
            route = [1, route_to[0]]
        self._spawn_vehicle(90, spawn_probability=1, go_straight=True,
                            position_deviation=0.1, velocity_deviation=0,
                            route=route)

        # Ego-vehicle
        MDPVehicle.SPEED_MIN = 0
        MDPVehicle.SPEED_MAX = 9
        MDPVehicle.SPEED_COUNT = 3
        # MDPVehicle.TAU_A = 1.0
        ego_lane = self.road.network.get_lane(("o0", "ir0", 0))
        destination = self.config["destination"] or "o" + str(self.np_random.randint(1, 4))
        # ego_vehicle = Vehicle(self.road, ego_lane.position(100, 0),
        #                       velocity=ego_lane.speed_limit*0.0,
        #                       heading=ego_lane.heading_at(50))
        ego_vehicle = ControlledLowLevelVehicle(self.road,
                                                ego_lane.position(95, 0),
                                                velocity=ego_lane.speed_limit,
                                                heading=ego_lane.heading_at(50)).plan_route_to(destination)
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        for v in self.road.vehicles:  # Prevent early collisions
            if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 20:
                self.road.vehicles.remove(v)

    def _spawn_vehicle(self,
                       longitudinal=70,
                       position_deviation=1.,
                       velocity_deviation=1.,
                       spawn_probability=0.6,
                       go_straight=False,
                       route=None,
                       velocity_init=8):
        if self.np_random.rand() > spawn_probability:
            return
        if not route:
            route = self.np_random.choice(range(4), size=2, replace=False)
            route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        print('spawn vehicle route ', route)
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(self.road, ("o" + str(route[0]), "ir" + str(route[0]), 0),
                                            longitudinal=longitudinal + 5 + self.np_random.randn() * position_deviation,
                                            velocity=velocity_init + self.np_random.randn() * velocity_deviation)
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self):
        is_leaving = lambda vehicle: "il" in vehicle.lane_index[0] and "o" in vehicle.lane_index[1] \
                                     and vehicle.lane.local_coordinates(vehicle.position)[0] \
                                     >= vehicle.lane.length - 4 * vehicle.LENGTH
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle is self.vehicle or not (is_leaving(vehicle) or vehicle.route is None)]

    @property
    def has_arrived(self):
        return "il" in self.vehicle.lane_index[0] \
               and "o" in self.vehicle.lane_index[1] \
               and self.vehicle.lane.local_coordinates(self.vehicle.position)[0] >= \
               self.vehicle.lane.length - 3 * self.vehicle.LENGTH

    def _cost(self, action):
        """
            The constraint signal is the occurence of collisions.
        """
        return float(self.vehicle.crashed)


class IntersectionLowLevelControlEnv12(IntersectionLowLevelControlEnv):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 2,
                "features": ["presence", "id", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-40, 40],
                    "y": [-40, 40],
                    "vx": [-40, 40],
                    "vy": [-40, 40],
                },
                "absolute": False,
                "order": "notsorted",
                "flatten": False,
                "observe_intentions": False
            },
            "duration": 4,  # [s]
            "policy_frequency": 10,  # [Hz]
            "destination": "o1",
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5 * 1.3,
            "collision_reward": IntersectionLowLevelControlEnv.COLLISION_REWARD,
            "normalize_reward": False,
            "range": 40,
            "num_vehicles": 0,
            "spawn_frequency": 10,
            "spawn": False,
            "route": [1, 2]
        })
        return config

    def _make_vehicles(self, n_vehicles=10):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 2  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3

        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [(self.road.act(), self.road.step(1 / self.SIMULATION_FREQUENCY)) for _ in range(self.SIMULATION_FREQUENCY)]

        # Challenger vehicle
        route = self.config['route']
        if self.config['route'] is None:
            route_to = self.np_random.choice([0,2,3], size=1, replace=False)
            route = [1, route_to[0]]
        self._spawn_vehicle(90, spawn_probability=1, go_straight=True,
                            position_deviation=0, velocity_deviation=0,
                            route=route, velocity_init=15)

        # Ego-vehicle
        MDPVehicle.SPEED_MIN = 0
        MDPVehicle.SPEED_MAX = 9
        MDPVehicle.SPEED_COUNT = 3
        # MDPVehicle.TAU_A = 1.0
        ego_lane = self.road.network.get_lane(("o0", "ir0", 0))
        destination = self.config["destination"] or "o" + str(self.np_random.randint(1, 4))
        # ego_vehicle = Vehicle(self.road, ego_lane.position(100, 0),
        #                       velocity=ego_lane.speed_limit*0.0,
        #                       heading=ego_lane.heading_at(50))
        ego_vehicle = ControlledLowLevelVehicle(self.road,
                                                ego_lane.position(100, 0),
                                                velocity=ego_lane.speed_limit,
                                                heading=ego_lane.heading_at(50)).plan_route_to(destination)
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        for v in self.road.vehicles:  # Prevent early collisions
            if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 20:
                self.road.vehicles.remove(v)


class IntersectionLowLevelControlEnv13(IntersectionLowLevelControlEnv):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 2,
                "features": ["presence", "id", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-40, 40],
                    "y": [-40, 40],
                    "vx": [-70, 70],
                    "vy": [-70, 70],
                },
                "absolute": False,
                "order": "notsorted",
                "flatten": False,
                "observe_intentions": False
            },
            "duration": 4,  # [s]
            "policy_frequency": 10,  # [Hz]
            "destination": "o1",
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5 * 1.3,
            "collision_reward": IntersectionLowLevelControlEnv.COLLISION_REWARD,
            "normalize_reward": False,
            "range": 40,
            "num_vehicles": 0,
            "spawn_frequency": 10,
            "spawn": False,
            "route": [1, 3]
        })
        return config


class IntersectionLowLevelControlEnv10(IntersectionLowLevelControlEnv):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 2,
                "features": ["presence", "id", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-40, 40],
                    "y": [-40, 40],
                    "vx": [-70, 70],
                    "vy": [-70, 70],
                },
                "absolute": False,
                "order": "notsorted",
                "flatten": False,
                "observe_intentions": False
            },
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "duration": 4,  # [s]
            "policy_frequency": 10,  # [Hz]
            "destination": "o1",
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5 * 1.3,
            "collision_reward": IntersectionLowLevelControlEnv.COLLISION_REWARD,
            "normalize_reward": False,
            "range": 40,
            "num_vehicles": 0,
            "spawn_frequency": 10,
            "spawn": False,
            "route": [1, 0]
        })
        return config


class IntersectionLowLevelControlEnv20(IntersectionLowLevelControlEnv):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 2,
                "features": ["presence", "id", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-40, 40],
                    "y": [-40, 40],
                    "vx": [-70, 70],
                    "vy": [-70, 70],
                },
                "absolute": False,
                "order": "notsorted",
                "flatten": False,
                "observe_intentions": False
            },
            "duration": 4,  # [s]
            "policy_frequency": 10,  # [Hz]
            "destination": "o1",
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5 * 1.3,
            "collision_reward": IntersectionLowLevelControlEnv.COLLISION_REWARD,
            "normalize_reward": False,
            "range": 40,
            "num_vehicles": 0,
            "spawn_frequency": 10,
            "spawn": False,
            "route": [2, 0]
        })
        return config

    def _make_vehicles(self, n_vehicles=10):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 2  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3

        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [(self.road.act(), self.road.step(1 / self.SIMULATION_FREQUENCY)) for _ in range(self.SIMULATION_FREQUENCY)]

        # Challenger vehicle
        route = self.config['route']
        if self.config['route'] is None:
            route_to = self.np_random.choice([0,2,3], size=1, replace=False)
            route = [1, route_to[0]]
        self._spawn_vehicle(97, spawn_probability=1, go_straight=True,
                            position_deviation=0, velocity_deviation=0,
                            route=route, velocity_init=8)

        # Ego-vehicle
        MDPVehicle.SPEED_MIN = 0
        MDPVehicle.SPEED_MAX = 9
        MDPVehicle.SPEED_COUNT = 3
        # MDPVehicle.TAU_A = 1.0
        ego_lane = self.road.network.get_lane(("o0", "ir0", 0))
        destination = self.config["destination"] or "o" + str(self.np_random.randint(1, 4))
        # ego_vehicle = Vehicle(self.road, ego_lane.position(100, 0),
        #                       velocity=ego_lane.speed_limit*0.0,
        #                       heading=ego_lane.heading_at(50))
        ego_vehicle = ControlledLowLevelVehicle(self.road,
                                                ego_lane.position(100, 0),
                                                velocity=ego_lane.speed_limit,
                                                heading=ego_lane.heading_at(50)).plan_route_to(destination)
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        for v in self.road.vehicles:  # Prevent early collisions
            if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 20:
                self.road.vehicles.remove(v)


class IntersectionLowLevelControlEnvMulti20(IntersectionLowLevelControlEnv):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 2,
                "features": ["presence", "id", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-40, 40],
                    "y": [-40, 40],
                    "vx": [-70, 70],
                    "vy": [-70, 70],
                },
                "absolute": False,
                "order": "notsorted",
                "flatten": False,
                "observe_intentions": False
            },
            "duration": 4,  # [s]
            "policy_frequency": 10,  # [Hz]
            "destination": "o1",
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5 * 1.3,
            "collision_reward": IntersectionLowLevelControlEnv.COLLISION_REWARD,
            "normalize_reward": False,
            "range": 40,
            "num_vehicles": 0,
            "spawn_frequency": 10,
            "spawn": False,
            "route": [2, 0],
            "challenge_vel": 2,
        })
        return config

    def _make_vehicles(self, n_vehicles=10):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 2  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3

        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [(self.road.act(), self.road.step(1 / self.SIMULATION_FREQUENCY)) for _ in range(self.SIMULATION_FREQUENCY)]

        # Challenger vehicle
        route = self.config['route']
        if self.config['route'] is None:
            route_to = self.np_random.choice([0,2,3], size=1, replace=False)
            route = [1, route_to[0]]

        for i in range(self.config["challenge_vel"]):
            self._spawn_vehicle(97 - 20*i, spawn_probability=1, go_straight=True,
                                position_deviation=0, velocity_deviation=0,
                                route=route, velocity_init=8)

        # Ego-vehicle
        MDPVehicle.SPEED_MIN = 0
        MDPVehicle.SPEED_MAX = 9
        MDPVehicle.SPEED_COUNT = 3
        # MDPVehicle.TAU_A = 1.0
        ego_lane = self.road.network.get_lane(("o0", "ir0", 0))
        destination = self.config["destination"] or "o" + str(self.np_random.randint(1, 4))
        # ego_vehicle = Vehicle(self.road, ego_lane.position(100, 0),
        #                       velocity=ego_lane.speed_limit*0.0,
        #                       heading=ego_lane.heading_at(50))
        ego_vehicle = ControlledLowLevelVehicle(self.road,
                                                ego_lane.position(100, 0),
                                                velocity=ego_lane.speed_limit,
                                                heading=ego_lane.heading_at(50)).plan_route_to(destination)
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        for v in self.road.vehicles:  # Prevent early collisions
            if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 20:
                self.road.vehicles.remove(v)


class IntersectionLowLevelControlEnv21(IntersectionLowLevelControlEnv):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 2,
                "features": ["presence", "id", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-40, 40],
                    "y": [-40, 40],
                    "vx": [-70, 70],
                    "vy": [-70, 70],
                },
                "absolute": False,
                "order": "notsorted",
                "flatten": False,
                "observe_intentions": False
            },
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "duration": 4,  # [s]
            "policy_frequency": 10,  # [Hz]
            "destination": "o1",
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5 * 1.3,
            "collision_reward": IntersectionLowLevelControlEnv.COLLISION_REWARD,
            "normalize_reward": False,
            "range": 40,
            "num_vehicles": 0,
            "spawn_frequency": 10,
            "spawn": False,
            "route": [2, 1]
        })
        return config

    def _make_vehicles(self, n_vehicles=10):
        """
            Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 2  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3

        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [(self.road.act(), self.road.step(1 / self.SIMULATION_FREQUENCY)) for _ in range(self.SIMULATION_FREQUENCY)]

        # Challenger vehicle
        route = self.config['route']
        if self.config['route'] is None:
            route_to = self.np_random.choice([0,2,3], size=1, replace=False)
            route = [1, route_to[0]]
        self._spawn_vehicle(94, spawn_probability=1, go_straight=True,
                            position_deviation=0, velocity_deviation=0,
                            route=route, velocity_init=15)

        # Ego-vehicle
        MDPVehicle.SPEED_MIN = 0
        MDPVehicle.SPEED_MAX = 9
        MDPVehicle.SPEED_COUNT = 3
        # MDPVehicle.TAU_A = 1.0
        ego_lane = self.road.network.get_lane(("o0", "ir0", 0))
        destination = self.config["destination"] or "o" + str(self.np_random.randint(1, 4))
        # ego_vehicle = Vehicle(self.road, ego_lane.position(100, 0),
        #                       velocity=ego_lane.speed_limit*0.0,
        #                       heading=ego_lane.heading_at(50))
        ego_vehicle = ControlledLowLevelVehicle(self.road,
                                                ego_lane.position(100, 0),
                                                velocity=ego_lane.speed_limit,
                                                heading=ego_lane.heading_at(50)).plan_route_to(destination)
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        for v in self.road.vehicles:  # Prevent early collisions
            if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 20:
                self.road.vehicles.remove(v)

register(
    id='intersection-v0',
    entry_point='highway_env.envs:IntersectionEnv',
)

register(
    id='intersection-v12',
    entry_point='highway_env.envs:IntersectionLowLevelControlEnv12',
)

register(
    id='intersection-v13',
    entry_point='highway_env.envs:IntersectionLowLevelControlEnv13',
)

register(
    id='intersection-v10',
    entry_point='highway_env.envs:IntersectionLowLevelControlEnv10',
)

register(
    id='intersection-v20',
    entry_point='highway_env.envs:IntersectionLowLevelControlEnv20',
)

register(
    id='intersectionMultiVehicle-v20',
    entry_point='highway_env.envs:IntersectionLowLevelControlEnvMulti20',
)

register(
    id='intersection-v21',
    entry_point='highway_env.envs:IntersectionLowLevelControlEnv21',
)