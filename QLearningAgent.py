from agent import HideAndSeekAgent
import numpy as np


class QLearningAgent(HideAndSeekAgent):

    def __init__(self, agent_type,num_actions, step_size, initial_epsilon, epsilon_decay, final_epsilon, discount_factor):
        # Initialize the agent with all parameters from the super class
        # This should allow us to call all functions of the superclass
        super().__init__(agent_type,num_actions, step_size, initial_epsilon, epsilon_decay, final_epsilon, discount_factor)

    def update(self, state: tuple[int,int], action: int, reward: float, terminated: bool,next_state: tuple[int,int]):
        """
        :param state: the state/observation returned after the last action taken
        :param action: the action chosen
        :param reward: the reward received after the last action taken
        :param terminated: has the episode terminated after taking the action in the state
        :param next_state: to what state have we transitioned?
        :return: chosen action
        """

        state, next_state = self.process_state(state, update = True), self.process_state(next_state, update = True)
        if not terminated:
            new_q_value = np.max(self.q_values[next_state])
        else:
            new_q_value = 0 # q-value of terminated state should be zero

        self.q_values[state][action] += self.step_size * (reward + self.discount_factor * new_q_value
                                                          - self.q_values[state][action])
