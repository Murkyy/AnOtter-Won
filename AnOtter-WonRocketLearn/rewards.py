import numpy as np
from typing import Tuple, List, Optional, Union
from rlgym.utils.common_values import BALL_RADIUS
from constants import FRAME_SKIP

from rlgym.utils import math
from rlgym.utils.common_values import (
    BLUE_TEAM,
    BLUE_GOAL_BACK,
    ORANGE_GOAL_BACK,
    ORANGE_TEAM,
    BALL_MAX_SPEED,
    CAR_MAX_SPEED,
)
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import (
    VelocityPlayerToBallReward,
)


class CombinedRewardNormalized(RewardFunction):
    """
    A reward composed of multiple rewards.
    """

    def __init__(
        self,
        reward_functions: Tuple[RewardFunction, ...],
        reward_weights: Optional[Tuple[float, ...]] = None,
    ):
        """
        Creates the combined reward using multiple rewards, and a potential set
        of weights for each reward.

        :param reward_functions: Each individual reward function.
        :param reward_weights: The weights for each reward.
        """
        super().__init__()

        self.reward_functions = reward_functions
        self.reward_weights = reward_weights or np.ones_like(reward_functions)

        if len(self.reward_functions) != len(self.reward_weights):
            raise ValueError(
                (
                    "Reward functions list length ({0}) and reward weights "
                    "length ({1}) must be equal"
                ).format(len(self.reward_functions), len(self.reward_weights))
            )

    @classmethod
    def from_zipped(
        cls, *rewards_and_weights: Union[RewardFunction, Tuple[RewardFunction, float]]
    ) -> "CombinedReward":
        """
        Alternate constructor which takes any number of either rewards, or (reward, weight) tuples.

        :param rewards_and_weights: a sequence of RewardFunction or (RewardFunction, weight) tuples
        """
        rewards = []
        weights = []
        for value in rewards_and_weights:
            if isinstance(value, tuple):
                r, w = value
            else:
                r, w = value, 1.0
            rewards.append(r)
            weights.append(w)
        return cls(tuple(rewards), tuple(weights))

    def pre_step(self, state: GameState) -> None:
        """
        Function to pre-compute values each step. This function is called only once each step, before get_reward is
        called for each player.J

        :param state: The current state of the game.
        """
        for func in self.reward_functions:
            func.pre_step(state)

    def reset(self, initial_state: GameState) -> None:
        """
        Resets underlying reward functions.

        :param initial_state: The initial state of the reset environment.
        """
        for func in self.reward_functions:
            func.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        """
        Returns the reward for a player on the terminal state.

        :param player: Player to compute the reward for.
        :param state: The current state of the game.
        :param previous_action: The action taken at the previous environment step.

        :return: The combined rewards for the player on the state.
        """
        rewards = [
            func.get_reward(player, state, previous_action)
            for func in self.reward_functions
        ]
        # if player.car_id == 1 or player.car_id == 2 :
        #     if ((self.reward_weights * np.array(rewards) / sum(self.reward_weights)) != [0., 0.]).any():
        #         print(
        #             "player",
        #             player.car_id,
        #             "combined_normed",
        #             self.reward_weights * np.array(rewards) / sum(self.reward_weights),
        #             "total",
        #             float(np.dot(self.reward_weights, rewards) / sum(self.reward_weights)),
        #         )

        return float(np.dot(self.reward_weights, rewards) / sum(self.reward_weights))

    def get_final_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        """
        Returns the reward for a player on the terminal state.

        :param player: Player to compute the reward for.
        :param state: The current state of the game.
        :param previous_action: The action taken at the previous environment step.

        :return: The combined rewards for the player on the state.
        """
        rewards = [
            func.get_final_reward(player, state, previous_action)
            for func in self.reward_functions
        ]

        return float(np.dot(self.reward_weights, rewards) / sum(self.reward_weights))


class EventReward(RewardFunction):
    def __init__(
        self,
        goal=0.0,
        team_goal=0.0,
        concede=-0.0,
        touch=0.0,
        shot=0.0,
        save=0.0,
        demo=0.0,
        boost_pickup=0.0,
    ):
        """
        :param goal: reward for goal scored by player.
        :param team_goal: reward for goal scored by player's team.
        :param concede: reward for goal scored by opponents. Should be negative if used as punishment.
        :param touch: reward for touching the ball.
        :param shot: reward for shooting the ball (as detected by Rocket League).
        :param save: reward for saving the ball (as detected by Rocket League).
        :param demo: reward for demolishing a player.
        :param boost_pickup: reward for picking up boost. big pad = +1.0 boost, small pad = +0.12 boost.
        """
        super().__init__()
        self.weights = np.array(
            [goal, team_goal, concede, touch, shot, save, demo, boost_pickup]
        )

        # Need to keep track of last registered value to detect changes
        self.last_registered_values = {}

    @staticmethod
    def _extract_values(player: PlayerData, state: GameState):
        if player.team_num == BLUE_TEAM:
            team, opponent = state.blue_score, state.orange_score
        else:
            team, opponent = state.orange_score, state.blue_score

        return np.array(
            [
                player.match_goals,
                team,
                opponent,
                player.ball_touched,
                player.match_shots,
                player.match_saves,
                player.match_demolishes,
                player.boost_amount,
            ]
        )

    def reset(self, initial_state: GameState, optional_data=None):
        # Update every reset since rocket league may crash and be restarted with clean values
        self.last_registered_values = {}
        for player in initial_state.players:
            self.last_registered_values[player.car_id] = self._extract_values(
                player, initial_state
            )

    def get_reward(
        self,
        player: PlayerData,
        state: GameState,
        previous_action: np.ndarray,
        optional_data=None,
    ):
        old_values = self.last_registered_values[player.car_id]
        new_values = self._extract_values(player, state)

        diff_values = new_values - old_values
        diff_values[diff_values < 0] = 0  # We only care about increasing values

        reward = np.dot(self.weights, diff_values)
        # norm_reward = reward / (sum(self.weights))
        self.last_registered_values[player.car_id] = new_values
        # if (reward != 0):
        #     print("Player ", player.car_id, "EventReward", reward )
        return reward


def _closest_to_ball(state: GameState) -> Tuple[int, int]:
    # returns [blue_closest, orange_closest]
    length = len(state.players)
    dist_list: List[float] = [100_000] * length
    blue_closest = -1
    orange_closest = -1
    blue_closest_id = -1
    orange_closest_id = -1

    for i, player in enumerate(state.players):
        # print("player", player.car_id, 'team', player.team_num)
        dist = np.linalg.norm(player.car_data.position - state.ball.position)
        dist_list[i] = dist
        if state.players[i].team_num == BLUE_TEAM and blue_closest == -1:
            blue_closest = i  # 1 2 5 6 for 2s | 1 2 3 5 6 7 for 3s
            blue_closest_id = state.players[i].car_id
        elif state.players[i].team_num == ORANGE_TEAM and orange_closest == -1:
            orange_closest = i
            orange_closest_id = state.players[i].car_id
        elif state.players[i].team_num == BLUE_TEAM and dist <= dist_list[blue_closest]:
            if dist == dist_list[blue_closest]:
                if (
                    state.players[i].car_data.position[0]
                    > state.players[blue_closest].car_data.position[0]
                ):
                    blue_closest = i
                    blue_closest_id = state.players[i].car_id
                    continue
            else:
                blue_closest = i
                blue_closest_id = state.players[i].car_id
                continue
        elif (
            state.players[i].team_num == ORANGE_TEAM
            and dist <= dist_list[orange_closest]
        ):
            if dist == dist_list[orange_closest]:
                if (
                    state.players[i].car_data.position[0]
                    < state.players[orange_closest].car_data.position[0]
                ):
                    orange_closest = i
                    orange_closest_id = state.players[i].car_id
                    continue
            else:
                orange_closest = i
                orange_closest_id = state.players[i].car_id
                continue
    return blue_closest_id, orange_closest_id


class KickoffReward(RewardFunction):
    """
    a simple reward that encourages driving towards the ball fast while it's in the neutral kickoff position
    """

    def __init__(self, kickoff_w=0.0):
        super().__init__()
        self.closest_reset_blue = -1
        self.closest_reset_orange = -1
        self.kickoff_timer = 0
        self.kickoff_timeout = 5 * 120 // FRAME_SKIP
        self.vel_dir_reward = VelocityPlayerToBallReward()
        self.kickoff_weight = kickoff_w

    def reset(self, initial_state: GameState):
        self.closest_reset_blue, self.closest_reset_orange = _closest_to_ball(
            initial_state
        )
        self.kickoff_timer = 0

    def pre_step(self, state: GameState):
        self.kickoff_timer += 1

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if (
            state.ball.position[0] == 0
            and state.ball.position[1] == 0
            and (
                self.closest_reset_blue
                == player.car_id  # 1 2 5 6 for 2s | 1 2 3 5 6 7 for 3s
                or self.closest_reset_orange == player.car_id
            )
            and self.kickoff_timer < self.kickoff_timeout
        ):
            reward += (
                self.vel_dir_reward.get_reward(player, state, previous_action)
                * self.kickoff_weight
            )
        return reward


class TouchGrassReward(RewardFunction):
    def __init__(self, touch_grass_w: float = 0.01) -> None:
        super().__init__()
        self.touch_grass_w = touch_grass_w

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        car_height = player.car_data.position[2]
        if player.on_ground and car_height < BALL_RADIUS:
            reward = -self.touch_grass_w
        return reward


class PossessionReward(RewardFunction):
    def __init__(self, possession_w: float = 1.0) -> None:
        super().__init__()
        self.possession_w = possession_w
        self.rewards_list: Tuple(float) = [0.0] * 2
        self.possession: Tuple(bool) = [False] * 2
        self.last_possession: Tuple(bool) = [False] * 2
        self.possession_changed: bool = False

    def reset(self, initial_state: GameState):
        self.last_possession = [False] * 2
        pass

    def pre_step(self, state: GameState):
        self.rewards_list = [0.0] * 2
        self.possession = [False] * 2
        self.possession_changed = False

        for player in state.players:
            if player.ball_touched:
                self.rewards_list[player.team_num] += self.possession_w
                self.rewards_list[1 - player.team_num] -= self.possession_w
                # print(f"player{player.car_id} touched the ball, reward list is {self.rewards_list}")
            if state.last_touch == player.car_id:
                self.possession[player.team_num] = True
                self.possession_changed = self.possession != self.last_possession
                # print(f"possession is {self.possession}, last possession is {self.last_possession}, possession_changed is {self.possession_changed}, " )
                if self.last_possession == [False] * 2:
                    # print(f"FIRST TOUCH, NO REWARD")
                    self.possession_changed = False
                self.last_possession = self.possession

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if self.possession_changed:
            # if self.rewards_list[1 - player.team_num] != 0.:
            # print("REWARD :", self.rewards_list[player.team_num], "PLAYER :",player.car_id )
            return self.rewards_list[player.team_num]
        return 0.0

class VelocityBallToGoalReward(RewardFunction):
    def __init__(self, own_goal=False, use_scalar_projection=False):
        super().__init__()
        self.own_goal = own_goal
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM and not self.own_goal \
                or player.team_num == ORANGE_TEAM and self.own_goal:
            objective = np.array(ORANGE_GOAL_BACK)
        else:
            objective = np.array(BLUE_GOAL_BACK)

        vel = state.ball.linear_velocity
        pos_diff = objective - state.ball.position
        if self.use_scalar_projection:
            # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
            # Max value should be max_speed / ball_radius = 2300 / 94 = 24.5
            # Used to guide the agent towards the ball
            inv_t = math.scalar_projection(vel, pos_diff)
            return inv_t
        else:
            # Regular component velocity
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            norm_vel = vel / BALL_MAX_SPEED
            # if (player.car_id == 1):
            #     print("Player ", player.car_id, "VelocityBallToGoalReward", float(np.dot(norm_pos_diff, norm_vel)))
            return float(np.dot(norm_pos_diff, norm_vel))