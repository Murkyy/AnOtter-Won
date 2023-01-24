from unittest import expectedFailure
from prettytable import PrettyTable
import numpy as np
from rlgym.envs import Match
from rlgym.utils.action_parsers import ContinuousAction, DiscreteAction
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import (
    VelocityBallToGoalReward,
)
from rlgym.utils.reward_functions.common_rewards.conditional_rewards import (
    RewardIfBehindBall,
    RewardIfTouchedLast,
)
from rlgym.utils.reward_functions.common_rewards.misc_rewards import (
    EventReward,
    SaveBoostReward,
    VelocityReward,
)
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import (
    FaceBallReward,
    LiuDistancePlayerToBallReward,
    TouchBallReward,
    VelocityPlayerToBallReward,
)
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import (
    GoalScoredCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
)
from rlgym_tools.extra_action_parsers.lookup_act import LookupAction
from rlgym_tools.extra_obs.general_stacking import GeneralStacker
from rlgym_tools.extra_rewards.distribute_rewards import DistributeRewards
from rlgym_tools.extra_rewards.kickoff_reward import KickoffReward
from rlgym_tools.extra_rewards.jump_touch_reward import JumpTouchReward
from rlgym_tools.extra_rewards.teamspacingreward import TeamSpacingReward
from rlgym_tools.extra_rewards.touchgrassreward import TouchGrassReward
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward, SB3CombinedLogRewardCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecCheckNan, VecMonitor, VecNormalize
from stable_baselines3.ppo import MlpPolicy
from torchsummary import summary

if __name__ == "__main__":  # Required for multiprocessing
    frame_skip = 8  # Number of ticks to repeat an action
    half_life_seconds = 10  # Easier to conceptualize, after this many seconds the reward discount is 0.5 # noqa: E501
    learning_rate = 1e-4
    horizon = 1_000_000
    wait_time = 20
    device = "cuda"
    savepath = "out/models/NoMad_Vbase/"
    logpath = "out/logs/NoMad_Vbase/"
    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs
    print(f"fps={fps}, gamma={gamma})")
    agents_per_match = 4
    num_instances = 12
    game_speed = 100


    def shift_bit_length(x):
        return 1 << (x - 1).bit_length()

    target_steps = shift_bit_length(horizon)
    steps = shift_bit_length(
        target_steps // (num_instances * agents_per_match)
    )  # making sure the experience counts line up properly
    batch_size = shift_bit_length(
        target_steps // 8
    )  # getting the batch size down to something more manageable - 100k in this case # noqa: E501
    training_interval = 50_000_000
    mmr_save_frequency = 100_000_000
    print(
        f"target_steps ={target_steps},n_steps ={steps}, batch_size={batch_size})"
    )  # noqa: E501

    def exit_save(model):
        model.save(savepath + "exit_save")

    def get_match():  # Need to use a function so that each instance can call it and produce their own objects # noqa: E501
        return Match(
            team_size=2,  # 3v3 to get as many agents going as possible, will make results more noisy # noqa: E501
            game_speed=game_speed,
            tick_skip=frame_skip,
            # SB3CombinedLogReward(CombinedReward
            # DistributeRewards(
            reward_function=SB3CombinedLogReward( #use no bumb reward and modify kickoff to single agent
                (
                    # LiuDistancePlayerToBallReward(),
                    # FaceBallReward(),
                    # TouchBallReward(aerial_weight=0.1),
                    # VelocityReward(),
                    # VelocityPlayerToBallReward(),
                    # SaveBoostReward(),
                    VelocityBallToGoalReward(),
                    EventReward(
                        goal=100.0,
                        concede=-100.0,
                        shot=5.0,
                        save=30.0,
                        demo=10.0,
                        boost_pickup=0.1,
                    ),
                    # RewardIfTouchedLast(),
                    # RewardIfBehindBall(),
                    # KickoffReward(),
                    # JumpTouchReward(),
                    # TeamSpacingReward(),
                    # TouchGrassReward(),
                ),
                (0.1, 1),
            ),
            # team_spirit=0.6,
            # ),
            # self_play=True,
            spawn_opponents=True,
            terminal_conditions=[
                TimeoutCondition(fps * 300),
                NoTouchTimeoutCondition(fps * 45),
                GoalScoredCondition(),
            ],
            # obs_builder = GeneralStacker(
            #     AdvancedObs(), stack_size=int(fps)
            # ),  # 1 sec stacking
            obs_builder = AdvancedObs(),
            state_setter=DefaultState(),  # Resets to kickoff position
            action_parser=LookupAction(),  # Discrete > Continuous don't @ me
        )

    env = SB3MultipleInstanceEnv(
        get_match, num_instances, wait_time=wait_time, force_paging=True
    )  # Start instances, waiting 60 seconds between each
    env = VecCheckNan(env)  # Optional
    env = VecMonitor(
        env
    )  # Recommended, logs mean reward and ep_len to Tensorboard # noqa: E501
    env = VecNormalize(
        env, norm_obs=False, norm_reward=True, gamma=gamma
    )  # Highly recommended, normalizes rewards

    try:
        
        model = PPO.load(
            savepath + "exit_save.zip",
            
            # savepath + "rl_model_225741420_steps.zip",
            env,
            custom_objects=dict(
                n_envs=env.num_envs,
                learning_rate=learning_rate,
                # _last_obs=None,  # noqa: E501
            ),  # Need this to change number of agents TODO : try to change lr or other parameters with custom_objects # noqa: E501
            device=device,  # Need to set device again (if using a specific one) # noqa: E501
        )
        print("Loaded previous exit save.", savepath + "exit_save.zip")

    except:
        print("No saved model found, creating new model.")
        # Hyperparameters presumably better than default; inspired by original PPO paper # noqa: E501
        from torch.nn import ReLU, Tanh

        policy_kwargs = dict(
            activation_fn=ReLU,
            net_arch=[
                512,
                dict(pi=[512, 512], vf=[512, 512]),
            ],
            # net_arch=[
            #     1024,
            #     1024,
            #     dict(pi=[1024, 1024, 1024, 1024], vf=[1024, 1024, 1024, 1024]),
            # ],
        )
        model = PPO(
            MlpPolicy,
            env,
            n_epochs=20,
            clip_range=0.2,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=4,
            gae_lambda=0.95,
            policy_kwargs=policy_kwargs,  # PPO calls for multiple epochs
            learning_rate=learning_rate,  # Around this is fairly common for PPO # noqa: E501
            ent_coef=0.01,  # From PPO Atari
            vf_coef=0.5,  # From PPO Atari
            gamma=gamma,  # Gamma as calculated using half-life
            verbose=2,  # Print out all the info as we're going
            batch_size=batch_size,  # Batch size as high as possible within reason # noqa: E501
            n_steps=steps,  # Number of steps to perform before optimizing network # noqa: E501
            tensorboard_log=logpath,  # `tensorboard --logdir out/logs` in terminal to see graphs # noqa: E501
            device=device,  # Uses GPU if available
        )

    print("layers sizes:")
    params = model.get_parameters()
    for k, v in params["policy"].items():
        print(k, list(v.size()))

    
    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step # noqa: E501
    # This saves to specified folder with a specified name
    callback = [CheckpointCallback(
        round(5_000_000 / env.num_envs),
        save_path=savepath,
        name_prefix="rl_model",  # noqa: E501
    ),
    SB3CombinedLogRewardCallback(("VelocityBallToGoalReward",
                    "EventReward"))
    ]
    model.num_timesteps = 0
    try:
        mmr_model_target_count = model.num_timesteps + mmr_save_frequency
        while True:
            # may need to reset timesteps when you're running a different number of instances than when you saved the model # noqa: E501
            model.learn(
                training_interval, callback=callback, reset_num_timesteps=False
            )  # can ignore callback if training_interval < callback target
            model.save(savepath + "exit_save")
            if model.num_timesteps >= mmr_model_target_count:
                model.save(f"out/mmr_models/{model.num_timesteps}")
                mmr_model_target_count += mmr_save_frequency
    except KeyboardInterrupt:
        print("Exiting training")

    print("Saving model")
    exit_save(model)
    print("Save complete")
