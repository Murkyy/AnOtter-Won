import os
import numpy as np
from rlgym.utils.action_parsers.discrete_act import DiscreteAction
from parsers.nomad_lookup_act import LookupAction
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class Agent:
    def __init__(self):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(cur_dir, "checkpoint.pt"), 'rb') as f:
            self.actor = torch.jit.load(f) 
        torch.set_num_threads(1)
        self.parser=LookupAction()


    def act(self, state):
        print("state print1\n", state, "")
        state = tuple(torch.from_numpy(np.asarray(s)).float() for s in state)
        with torch.no_grad():
            print("state print2\n", state, "dtype\n", type(state))
            out, weights = self.actor(state)
        self.state = state

        out = (out,)
        max_shape = max(o.shape[-1] for o in out)
        logits = torch.stack(
            [
                l
                if l.shape[-1] == max_shape
                else F.pad(l, pad=(0, max_shape - l.shape[-1]), value=float("-inf"))
                for l in out
            ],
            dim=1
        )
        dist = Categorical(logits=logits)
        actions = dist.sample()

        # print(Categorical(logits=logits).sample())
        x = self.parser.parse_actions(actions.numpy().item()[0], state)
        return x


if __name__ == "__main__":
    print("You're doing it wrong.")
