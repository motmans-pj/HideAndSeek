from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm
import gymnasium as gym

import random


class HideAndSeekAgent:
    """
    A superclass from which specific agents can inherit, specifically by changing the policy method
    """

    def __init__(self, agent_type, num_actions, step_size, initial_epsilon, epsilon_decay, final_epsilon,
                 discount_factor):
        """
        :param agent_type: is our agent a hider or a seeker?
        :param step_size: the learning rate used for training
        :param initial_epsilon: the initial epsilon for the epsilon greedy strategy
        :param epsilon_decay: how does epsilon decrease as the number of episodes increases
        :param final_epsilon: what epsilon to use after training
        :param discount_factor: trading off current and future rewards
        :param num_actions: how many actions can the agent take
        """

        # Type of agent
        self.agent_type = agent_type

        # How many actions can the agent choose from
        self.num_actions = num_actions

        # Learning rate for learning scheme
        self.step_size = step_size

        # Epsilon scheme for the epsilon greedy strategy
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Discount factor for the future rewards
        self.discount_factor = discount_factor

        # We don't know the number of states, how then do we initialize the q-values?
        # We first use a tabular solution method so still want to store q(s,a) for all s,a
        # for each new state we encounter, we will store a numpy array of length num_actions
        # In the end, we will then store a table of shape (no_of_states,no_of_actions)
        self.q_values = defaultdict(lambda: np.zeros(num_actions))

    def process_state(self, state, update=False, static=False):
        """
        :param state: a dictionary, for the seeker it is
        => {"Seeker_location":self.seeker_location,
        "Obstacles": self._is_obstacle,"LineOfSight": self.line_of_sight}

        What this function does is to keep unique states as (xseeker, yseeker, xhider, yhider)

        :return: a tuple of the dictionary's values that can serve as
        key for a dictionary in which we keep the q_values (see self.q_values)
        """
        if not update:
            if self.agent_type == "Seeker":
                state = state["Seeker"]
            elif self.agent_type == "Hider":
                state = state["Hider"]
        result = []
        result.append(state["Seeker_location"][:])
        result.append(state["Hider_location"][:])
        result = [item for sublist in result for item in sublist]
        return tuple(result)

        # if self.agent_type == "Seeker":
        # key = ' | '.join(str(lst) for lst in obs["Seeker"].values())
        # elif self.agent_type == "Hider":
        # key = ' | '.join(str(lst) for lst in obs["Hider"].values())

    def argmax(self, q_values, available_actions):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if i in available_actions:
                if q_values[i] > top:
                    top = q_values[i]
                    ties = []

                if q_values[i] == top:
                    ties.append(i)

        return random.choice(ties)

    def policy(self, state, available_actions) -> int:
        """
        Implementing the epsilon greedy strategy
        :param state: the state is a dictionary that depends on the type of agent
        :param available_actions: the actions that are available so that the
        agent stays in the grid and doesn't move towards an obstacle.
        :return: the chosen action
        """
        # Process the state into a string such that it can be used to store q_values
        state = self.process_state(state)
        if np.random.random() < self.epsilon:
            action = random.choice(available_actions)
        else:
            action = self.argmax(self.q_values[state], available_actions)
        return action

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)



