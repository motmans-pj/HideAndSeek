from agent import HideAndSeekAgent
import numpy as np


class ExpectedSarsaAgent(HideAndSeekAgent):

    def __init__(self, agent_type,num_actions, step_size, initial_epsilon, epsilon_decay, final_epsilon, discount_factor):
        # Initialize the agent with all parameters from the super class
        # This should allow us to call all functions of the superclass
        super().__init__(agent_type,num_actions, step_size, initial_epsilon, epsilon_decay, final_epsilon, discount_factor)

    def update(self, state: tuple, action: int, reward: float, terminated: bool, next_state: tuple):
        """
        saRSa, we have already done: first state to first action, received a reward and next state for that
        now we need to take an action and update
        :param state: the state/observation returned after the last action taken
        :param action: the action chosen
        :param reward: the reward received after the last action taken
        :param terminated: has the episode terminated after taking the action in the state
        :param next_state: to what state have we transitioned?
        :return: chosen action
        """
        state, next_state = self.process_state(state, update = True), self.process_state(next_state, update = True)
        # We attempt expected Sarsa
        current_q = self.q_values[next_state]
        # give epsilon/num_actions probability to all
        policy_new_state = self.epsilon/self.num_actions * np.ones(self.num_actions)
        # give more to greedy actions (1-epsilon)/no_of_optimal_actions
        policy_new_state[np.argmax(current_q)] += (1 - self.epsilon)/np.sum(current_q == np.max(current_q))
        # q_values is a dictionary: {state: np.array(x,y)} where x is the q_value for state and action 0, y for action 1
        self.q_values[state][action] += self.step_size * (reward +
                                                          self.discount_factor * np.dot(policy_new_state, current_q)
                                                          - self.q_values[state][action])
