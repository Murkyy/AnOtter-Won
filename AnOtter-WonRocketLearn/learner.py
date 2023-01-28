import os
import wandb
import numpy as np
from typing import Any
from prettytable import PrettyTable
from functools import partial
import random

import torch.jit
from torch.nn import Linear, Sequential, ReLU, Module, init

from redis import Redis

from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym.utils.gamestates import PlayerData, GameState

from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from rlgym_tools.extra_rewards.jump_touch_reward import JumpTouchReward
from rewards import (
    EventReward,
    KickoffReward,
    CombinedRewardNormalized,
    TouchGrassReward,
    PossessionReward,
    VelocityBallToGoalReward,
)
from rlgym.utils.action_parsers.discrete_act import DiscreteAction
from anotterwon_lookup_act import LookupAction


from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy

# from rocket_learn.ppo import PPO
from ppo import PPO
from rocket_learn.rollout_generator.redis.redis_rollout_generator import (
    RedisRolloutGenerator,
)
from rocket_learn.utils.util import SplitLayer

torch.manual_seed(2)
random.seed(2)
np.random.seed(2)

# ROCKET-LEARN ALWAYS EXPECTS A BATCH DIMENSION IN THE BUILT OBSERVATION
class ExpandAdvancedObs(AdvancedObs):
    def build_obs(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return np.expand_dims(obs, 0)


def shift_bit_length(x):
    return 1 << (x - 1).bit_length()


if __name__ == "__main__":
    """

    Starts up a rocket-learn learner process, which ingests incoming data, updates parameters
    based on results, and sends updated model parameters out to the workers

    """

    # ROCKET-LEARN USES WANDB WHICH REQUIRES A LOGIN TO USE. YOU CAN SET AN ENVIRONMENTAL VARIABLE
    # OR HARDCODE IT IF YOU ARE NOT SHARING YOUR SOURCE FILES
    name_and_version = "AnOtterWon_Vbase2"
    wandb.login(key=os.environ["wandb_key"])
    logger = wandb.init(project="AnOtter-Won", entity="murky")
    logger.name = "LEARNER_ANOTTERWON_Vbase2"

    # LINK TO THE REDIS SERVER YOU SHOULD HAVE RUNNING (USE THE SAME PASSWORD YOU SET IN THE REDIS
    # CONFIG)
    redis = Redis(password=os.environ["redis_password"])

    # ** ENSURE OBSERVATION, REWARD, AND ACTION CHOICES ARE THE SAME IN THE WORKER **
    def obs():
        return ExpandAdvancedObs()

    def rew():
        return CombinedRewardNormalized(
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
            (
                1,
                0.1,
                0.1,
                1,
                0.01
            ),
        )

    def act():
        return LookupAction()

    # THE ROLLOUT GENERATOR CAPTURES INCOMING DATA THROUGH REDIS AND PASSES IT TO THE LEARNER.
    # -save_every SPECIFIES HOW OFTEN REDIS DATABASE IS BACKED UP TO DISK
    # -model_every SPECIFIES HOW OFTEN OLD VERSIONS ARE SAVED TO REDIS. THESE ARE USED FOR TRUESKILL
    # COMPARISON AND TRAINING AGAINST PREVIOUS VERSIONS
    # -clear DELETE REDIS ENTRIES WHEN STARTING UP (SET TO FALSE TO CONTINUE WITH OLD AGENTS)
    rollout_gen = RedisRolloutGenerator(
        name_and_version,
        redis,
        obs,
        rew,
        act,
        logger=logger,
        save_every=100,
        model_every=100,
        clear=True,
    )

    # ROCKET-LEARN EXPECTS A SET OF DISTRIBUTIONS FOR EACH ACTION FROM THE NETWORK, NOT
    # THE ACTIONS THEMSELVES. SEE network_setup.readme.txt FOR MORE INFORMATION
    # split = (3, 3, 3, 3, 3, 2, 2, 2)
    split = (90,)
    total_output = sum(split)

    # TOTAL SIZE OF THE INPUT DATA

    state_dim = 169  # 107 for 1s, 169 for 2s, 231 for 3s for Advanced Obs

    hidden_dim = 256

    critic = Sequential(
        Linear(state_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, 1),
    )

    actor = DiscretePolicy(
        Sequential(
            Linear(state_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, total_output),
            SplitLayer(splits=split),
        ),
        split,
    )

    # CREATE THE OPTIMIZER
    optim = torch.optim.Adam(
        [
            {"params": actor.parameters(), "lr": 1e-4},
            {"params": critic.parameters(), "lr": 1e-4},
        ]
    )

    def init_weights(module: Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """

        if isinstance(module, (Linear)):
            if module.in_features == hidden_dim and module.out_features == 1:
                gain = 1
            if module.in_features == hidden_dim and module.out_features == hidden_dim:
                gain = np.sqrt(2)
            if module.in_features == hidden_dim and module.out_features == total_output:
                gain = 0.01
            init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    # PPO REQUIRES AN ACTOR/CRITIC AGENT
    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)
    agent.apply(partial(init_weights))  # Orthogonal weights init as in Atari

    # for name, parameter in agent.named_parameters():
    #     print(name)
    #     print(parameter.data)

    n_steps = shift_bit_length(1_000_000)
    batch_size = n_steps
    minibatch_size = shift_bit_length(100_000)
    tick_skip = 8
    fps = 120 / tick_skip
    half_life_seconds = 10
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))
    print(
        "gamma =",
        gamma,
        "n_steps =",
        n_steps,
        "batch_size =",
        batch_size,
        "minibatch_size =",
        minibatch_size,
    )
    # INSTANTIATE THE PPO TRAINING ALGORITHM
    alg = PPO(
        rollout_gen,
        agent,
        ent_coef=0.01,
        n_steps=n_steps,
        batch_size=batch_size,
        minibatch_size=minibatch_size,
        epochs=20,
        gamma=gamma,
        clip_range=0.2,
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,
        logger=logger,
        device="cuda",
    )

    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        critic_params = 0
        actor_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            if "critic" in name:
                critic_params += params
            if "actor" in name:
                actor_params += params
            total_params += params
        print(table)
        print(f"Actor Params: {actor_params}")
        print(f"Critic Params: {critic_params}")
        print(f"Total Trainable Params: {total_params}")
        return total_params

    print(count_parameters(actor))
    print(count_parameters(critic))

    # LOAD A CHECKPOINT THAT WAS PREVIOUSLY SAVED AND CONTINUE TRAINING. OPTIONAL PARAMETER ALLOWS YOU
    # TO RESTART THE STEP COUNT INSTEAD OF CONTINUING
    # alg.load(f"out/models/{name_and_version}/AnOtter-Won_1674253831.9620147/AnOtter-Won_100/checkpoint.pt")
    # BEGIN TRAINING. IT WILL CONTINUE UNTIL MANUALLY STOPPED
    # -iterations_per_save SPECIFIES HOW OFTEN CHECKPOINTS ARE SAVED
    # -save_dir SPECIFIES WHERE
    alg.run(
        iterations_per_save=100,
        save_dir=f"out/models/{name_and_version}",
        save_jit=True,
    )
