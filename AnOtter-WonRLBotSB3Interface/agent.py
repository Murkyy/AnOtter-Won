import numpy as np
from rocket_learn.ppo import PPO
import pathlib
from parsers.lookup_act import LookupAction


class Agent:
    def __init__(self):
        _path = pathlib.Path(__file__).parent.resolve()
        custom_objects = {
            "lr_schedule": 0.000001,
            "clip_range": .02,
            "n_envs": 1,
        }
        
        self.actor = PPO.load(str(_path) + '/exit_save', device='cuda', custom_objects=custom_objects)
        self.parser = LookupAction()


    def act(self, state):
        action = self.actor.predict(state, deterministic=True)
        x = self.parser.parse_actions(action[0], state)
        return x

if __name__ == "__main__":
    print("You're doing it wrong.")
