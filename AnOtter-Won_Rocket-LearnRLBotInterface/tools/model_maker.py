import torch
import os

from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from torch.nn import Linear, Sequential, ReLU
from rocket_learn.utils.util import SplitLayer


# TODO add your network here
# TOTAL SIZE OF THE INPUT DATA
state_dim = 169

hidden_dim = 256
split = (90,)
total_output = sum(split)

def get_actor(split, state_dim):
    return DiscretePolicy(Sequential(
        Linear(state_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, total_output),
        SplitLayer(splits=split)
    ), split)


actor = get_actor(split, state_dim)

# PPO REQUIRES AN ACTOR/CRITIC AGENT

cur_dir = os.path.dirname(os.path.realpath(__file__))
checkpoint = torch.load(os.path.join(cur_dir, "checkpoint.pt"))
actor.load_state_dict(checkpoint['actor_state_dict'])
actor.eval()
torch.jit.save(torch.jit.script(actor), 'jit.pt')

exit(0)
