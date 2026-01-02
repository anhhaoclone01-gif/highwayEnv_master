# highway_env.py (modified reward to include safety metrics)
from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


Observation = np.ndarray


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 4,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,
                "high_speed_reward": 0.4,
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
                # Safety-related config (weights default 0 for backward compatibility)
                "safety_weight": 0.0,
                "lane_change_danger_weight": 0.0,
                "t_safe": 2.0,
                "a_max_threshold": 6.0,
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25.0,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _compute_danger_metrics(self) -> dict:
        """
        Compute TTC/gap/rel_v/a_req/danger_flag/min_ttc for the ego vehicle by scanning other vehicles.

        Returns dict:
            {
                "ttc_front": float (inf if none),
                "gap_front": float,
                "rel_v_front": float,
                "a_req_front": float,
                "danger_flag": bool,
                "min_ttc": float (inf if none)
            }
        """
        ego = self.vehicle
        INF = float("inf")

        ttc_front = INF
        a_req_front = 0.0
        gap_front = INF
        rel_v_front = 0.0
        min_ttc = INF

        def longitudinal_distance(ego_v, other_v):
            delta = other_v.position - ego_v.position
            heading = np.array([np.cos(ego_v.heading), np.sin(ego_v.heading)])
            return float(np.dot(delta, heading))

        def longitudinal_rel_speed(ego_v, other_v):
            heading = np.array([np.cos(ego_v.heading), np.sin(ego_v.heading)])
            return float(np.dot(ego_v.velocity - other_v.velocity, heading))

        def compute_ttc_and_a_req(gap, rel_v, eps=1e-6):
            if gap <= 0:
                return 0.0, float("inf")
            if rel_v <= 0:
                return INF, 0.0
            ttc = gap / rel_v
            a_req = (rel_v ** 2) / (2.0 * gap) if gap > eps else float("inf")
            return float(ttc), float(a_req)

        for other in self.road.vehicles:
            if other is ego or not other.solid:
                continue

            longitudinal = longitudinal_distance(ego, other)
            rel_v = longitudinal_rel_speed(ego, other)

            if longitudinal > 0:
                gap = longitudinal - (other.LENGTH + ego.LENGTH) / 2.0
                ttc, a_req = compute_ttc_and_a_req(gap, rel_v)
                if ttc < ttc_front:
                    ttc_front = ttc
                    a_req_front = a_req
                    gap_front = max(gap, 0.0)
                    rel_v_front = rel_v
                # track neighbor ttc
                if rel_v > 0 and gap > 0:
                    ttc_neighbor = gap / rel_v
                    if 0.0 <= ttc_neighbor < min_ttc:
                        min_ttc = ttc_neighbor

        t_safe = float(self.config.get("t_safe", 2.0))
        a_max_threshold = float(self.config.get("a_max_threshold", 6.0))
        danger_flag = False
        if (ttc_front < t_safe) or (a_req_front > a_max_threshold):
            danger_flag = True

        return {
            "ttc_front": ttc_front,
            "gap_front": gap_front,
            "rel_v_front": rel_v_front,
            "a_req_front": a_req_front,
            "danger_flag": danger_flag,
            "min_ttc": min_ttc,
        }

    def _reward(self, action: Action) -> float:
        rewards = self._rewards(action)

        # Base reward (collision + right lane + speed)
        base_sum = (
            self.config.get("collision_reward", 0) * rewards.get("collision_reward", 0)
            + self.config.get("right_lane_reward", 0) * rewards.get("right_lane_reward", 0)
            + self.config.get("high_speed_reward", 0) * rewards.get("high_speed_reward", 0)
        )

        if self.config.get("normalize_reward", True):
            base_sum = utils.lmap(
                base_sum,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )

        final_reward = base_sum * rewards.get("on_road_reward", 1.0)

        # Add safety and lane-change penalties (weights configurable)
        safety_weight = float(self.config.get("safety_weight", 0.0))
        lane_change_danger_weight = float(self.config.get("lane_change_danger_weight", 0.0))

        final_reward += safety_weight * rewards.get("safety_reward", 0.0)
        final_reward += lane_change_danger_weight * rewards.get("lane_change_penalty", 0.0)

        return float(final_reward)

    def _rewards(self, action: Action) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )

        # Compute danger-based metrics and penalties
        danger = self._compute_danger_metrics()
        safety_reward = 0.0
        if np.isfinite(danger["ttc_front"]):
            t_safe = float(self.config.get("t_safe", 2.0))
            if danger["ttc_front"] < t_safe:
                safety_reward = -float((t_safe - danger["ttc_front"]) / max(t_safe, 1e-6))
        lane_change_penalty = 0.0
        try:
            if isinstance(action, (int, np.integer)):
                action_name = None
                if hasattr(self.action_type, "actions"):
                    action_name = self.action_type.actions.get(int(action), None)
                if action_name in ["LANE_LEFT", "LANE_RIGHT"] and danger["danger_flag"]:
                    lane_change_penalty = -1.0
        except Exception:
            lane_change_penalty = 0.0

        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
            "safety_reward": float(safety_reward),
            "lane_change_penalty": float(lane_change_penalty),
        }

    def _is_terminated(self) -> bool:
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]