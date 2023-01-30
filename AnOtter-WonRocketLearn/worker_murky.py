from typing import Any
import numpy
import os

from redis import Redis

from rlgym.envs import Match
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.terminal_conditions.common_conditions import (
    GoalScoredCondition,
    TimeoutCondition,
    NoTouchTimeoutCondition,
)

from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import (
    VelocityPlayerToBallReward,
)
from rlgym_tools.extra_rewards.jump_touch_reward import JumpTouchReward
from rewards import (
    EventReward,
    KickoffReward,
    CombinedRewardNormalized,
    TouchGrassReward,
    PossessionReward,
    VelocityBallToGoalReward,
)
from rlgym.utils.state_setters.default_state import DefaultState
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym.utils.action_parsers.discrete_act import DiscreteAction
from anotterwon_lookup_act import LookupAction


from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker


# ROCKET-LEARN ALWAYS EXPECTS A BATCH DIMENSION IN THE BUILT OBSERVATION
class ExpandAdvancedObs(AdvancedObs):
    def build_obs(
        self, player: PlayerData, state: GameState, previous_action: numpy.ndarray
    ) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)


if __name__ == "__main__":
    """

    Starts up a rocket-learn worker process, which plays out a game, sends back game data to the
    learner, and receives updated model parameters when available

    """

    # OPTIONAL ADDITION:
    # LIMIT TORCH THREADS TO 1 ON THE WORKERS TO LIMIT TOTAL RESOURCE USAGE
    # TRY WITH AND WITHOUT FOR YOUR SPECIFIC HARDWARE
    import torch

    torch.set_num_threads(1)

    frame_skip = 8
    fps = 120 / frame_skip
    # BUILD THE ROCKET LEAGUE MATCH THAT WILL USED FOR TRAINING
    # -ENSURE OBSERVATION, REWARD, AND ACTION CHOICES ARE THE SAME IN THE WORKER
    match = Match(
        game_speed=100,
        spawn_opponents=True,
        team_size=2,
        state_setter=DefaultState(),
        obs_builder=ExpandAdvancedObs(),
        action_parser=LookupAction(),
        terminal_conditions=[
            TimeoutCondition(round(4096)),
            NoTouchTimeoutCondition(fps * 45),
            GoalScoredCondition(),
        ],
        # mode | gpm  | 95% | 99%
        # -----|------|-----|-----
        # 1v1  | 1.69 | 1.8 | 2.8
        # 2v2  | 1.07 | 2.8 | 4.3
        # 3v3  | 0.83 | 3.6 | 5.6
        reward_function=CombinedRewardNormalized(
            (
                EventReward(
                    team_goal=10.0,
                    concede=-10.0,
                    shot=0.5,
                    save=3.0,
                    demo=1.0,
                    boost_pickup=0.01,
                ),
                KickoffReward(kickoff_w=1.0),
                VelocityBallToGoalReward(),
                JumpTouchReward(),
                VelocityPlayerToBallReward(),
            ),
            (1, 0.1, 0.1, 0.1, 0.01),
        ),
    )

    # LINK TO THE REDIS SERVER YOU SHOULD HAVE RUNNING (USE THE SAME PASSWORD YOU SET IN THE REDIS
    # CONFIG)
    r = Redis(host="127.0.0.1", password=os.environ["redis_password"])

    # LAUNCH ROCKET LEAGUE AND BEGIN TRAINING
    # -past_version_prob SPECIFIES HOW OFTEN OLD VERSIONS WILL BE RANDOMLY SELECTED AND TRAINED AGAINST
    RedisRolloutWorker(
        r,
        "Murky_PC",
        match,
        past_version_prob=0.2,
        evaluation_prob=0.01,
        sigma_target=2,
        dynamic_gm=False,
        send_obs=True,
        streamer_mode=False,
        send_gamestates=False,
        force_paging=False,
        auto_minimize=True,
        local_cache_name="Murky_PC_model_database",
    ).run()
