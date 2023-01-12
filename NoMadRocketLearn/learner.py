import os
import wandb
import numpy as np
from typing import Any
from prettytable import PrettyTable

import torch.jit
from torch.nn import Linear, Sequential, ReLU

from redis import Redis

from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions.default_reward import DefaultReward
from rewards import EventReward, KickoffReward, CombinedRewardNormalized, TouchGrassReward, TouchBallReward
from rlgym.utils.action_parsers.discrete_act import DiscreteAction
from nomad_lookup_act import LookupAction



from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.utils.util import SplitLayer



# ROCKET-LEARN ALWAYS EXPECTS A BATCH DIMENSION IN THE BUILT OBSERVATION
class ExpandAdvancedObs(AdvancedObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return np.expand_dims(obs, 0)


if __name__ == "__main__":
    """
    
    Starts up a rocket-learn learner process, which ingests incoming data, updates parameters
    based on results, and sends updated model parameters out to the workers
    
    """

    # ROCKET-LEARN USES WANDB WHICH REQUIRES A LOGIN TO USE. YOU CAN SET AN ENVIRONMENTAL VARIABLE
    # OR HARDCODE IT IF YOU ARE NOT SHARING YOUR SOURCE FILES
    name_and_version = "NoMad_V07"
    wandb.login(key=os.environ["wandb_key"])
    logger = wandb.init(project="nomad", entity="murky")
    logger.name = "LEARNER_NOMAD_V07"

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
                    goal=1.0,
                    concede=-1.0,
                    shot=0.05,
                    save=0.3,
                    demo=0.1,
                    boost_pickup=0.05,
                    touch=0.01
                ),
                KickoffReward(kickoff_w=1.0),
                TouchBallReward(touch_ball_w=1.0)
            ),
            (1, 1, 1, 1),
        )

    def act():
        return LookupAction()


    # THE ROLLOUT GENERATOR CAPTURES INCOMING DATA THROUGH REDIS AND PASSES IT TO THE LEARNER.
    # -save_every SPECIFIES HOW OFTEN REDIS DATABASE IS BACKED UP TO DISK
    # -model_every SPECIFIES HOW OFTEN OLD VERSIONS ARE SAVED TO REDIS. THESE ARE USED FOR TRUESKILL
    # COMPARISON AND TRAINING AGAINST PREVIOUS VERSIONS
    # -clear DELETE REDIS ENTRIES WHEN STARTING UP (SET TO FALSE TO CONTINUE WITH OLD AGENTS)
    rollout_gen = RedisRolloutGenerator(name_and_version, redis, obs, rew, act,
                                        logger=logger,
                                        save_every=100,
                                        model_every=100,
                                        clear=True)

    # ROCKET-LEARN EXPECTS A SET OF DISTRIBUTIONS FOR EACH ACTION FROM THE NETWORK, NOT
    # THE ACTIONS THEMSELVES. SEE network_setup.readme.txt FOR MORE INFORMATION
    # split = (3, 3, 3, 3, 3, 2, 2, 2)
    split = (90,)
    total_output = sum(split)

    # TOTAL SIZE OF THE INPUT DATA
    
    state_dim = 169 # 107 for 1s, 169 for 2s, 231 for 3s for Advanced Obs


    critic = Sequential(
        Linear(state_dim, 512),
        ReLU(),
        Linear(512, 512),
        ReLU(),
        Linear(512, 512),
        ReLU(),
        Linear(512, 1)
    )

    actor = DiscretePolicy(Sequential(
        Linear(state_dim, 512),
        ReLU(),
        Linear(512, 512),
        ReLU(),
        Linear(512, 512),
        ReLU(),
        Linear(512, total_output),
        SplitLayer(splits=split)
    ), split)

    # CREATE THE OPTIMIZER
    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": 5e-5},
        {"params": critic.parameters(), "lr": 5e-5}
    ])

    # PPO REQUIRES AN ACTOR/CRITIC AGENT
    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)

    tick_skip=8
    fps = 120/tick_skip
    half_life_seconds = 10
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))
    print('gamma =', gamma)
    # INSTANTIATE THE PPO TRAINING ALGORITHM
    alg = PPO(
        rollout_gen,
        agent,
        ent_coef=0.01,
        n_steps=400_000,
        batch_size=400_000,
        minibatch_size=100_000,
        epochs=30,
        gamma=gamma,
        clip_range=0.2,
        gae_lambda=0.95,
        vf_coef=0.75,
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
    # alg.load(f"out/models/{name_and_version}/nomad_latest/nomad_-1/checkpoint.pt")

    # BEGIN TRAINING. IT WILL CONTINUE UNTIL MANUALLY STOPPED
    # -iterations_per_save SPECIFIES HOW OFTEN CHECKPOINTS ARE SAVED
    # -save_dir SPECIFIES WHERE
    alg.run(iterations_per_save=100, save_dir=f"out/models/{name_and_version}", save_jit=True)
