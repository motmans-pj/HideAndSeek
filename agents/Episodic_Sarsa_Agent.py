from agents.agent import HideAndSeekAgent
import numpy as np
import random 
import utilities.tiles3 as tc

# TileCoder class for Hide & Seek environment
class TileCoder:
    def __init__(self, iht_size=32768, num_tilings=8, num_tiles=8):
        """
        Initializes the tile Coder
        Initializers:
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the
                     tile coder are the same
        Class Variables:
        self.iht -- tc.IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """
        self.iht = tc.IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles

    def get_tiles(self, x_seeker, y_seeker, x_hider, y_hider, size=8):
        """
        Takes in positions from hider and seeker and returns a numpy array of active tiles.

        Arguments:
        x_seeker -- int, the horizontal position of the agent between 0 and size - 1
        y_seeker -- int, the vertical position of the agent between 0 and size - 1
        x_hider -- int, the horizontal position of the agent between 0 and size - 1
        x_hider -- int, the vertical position of the agent between 0 and size - 1
        returns:
        tiles - np.array, active tiles
        """
        # Use the ranges above and self.num_tiles to scale positions to the range [0, 1]
        # then multiply that range with self.num_tiles so it scales from [0, num_tiles]

        # range
        min = 0
        max = size-1
        # scaled
        x_seeker_scaled = 0
        y_seeker_scaled = 0
        x_hider_scaled = 0
        y_hider_scaled = 0

        x_seeker_scaled = (x_seeker - min) / (max - min) * self.num_tiles
        y_seeker_scaled = (y_seeker - min) / (max - min) * self.num_tiles
        x_hider_scaled = (x_hider - min) / (max - min) * self.num_tiles
        y_hider_scaled = (y_hider - min) / (max - min) * self.num_tiles  

        # get the tiles using tc.tiles, with self.iht, self.num_tilings and [scaled position, scaled velocity]
        # nothing to implment here:
        tiles = tc.tiles(self.iht, self.num_tilings, [x_seeker_scaled, y_seeker_scaled, x_hider_scaled, y_hider_scaled])

        return np.array(tiles)

class EpisodicSemiGradientSarsaAgent(HideAndSeekAgent):
    """
    Initialization of EpisodicSemiGradientSarsa Agent. See algorithm p244 book
    All values are set to None so they can
    be initialized in the agent_init method.
    """
    def __init__(self, agent_type,num_actions, step_size, initial_epsilon, epsilon_decay, final_epsilon, discount_factor):
        # Initialize the agent with all parameters from the super class
        # This should allow us to call all functions of the superclass
        super().__init__(agent_type,num_actions, step_size, initial_epsilon, epsilon_decay, final_epsilon, discount_factor)

        self.iht_size = None
        self.w = None
        self.num_tilings = None
        self.num_tiles = None
        self.fbtc = None
        self.initial_weights = None
        self.previous_tiles = None
        self.epsilon = initial_epsilon

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""
        self.num_tilings = agent_info.get("num_tilings", 8)
        self.num_tiles = agent_info.get("num_tiles", 8)
        self.iht_size = agent_info.get("iht_size", 32768)
        self.step_size = self.step_size / self.num_tilings
        self.initial_weights = agent_info.get("initial_weights", 0.0)
        
        # We initialize self.w to two times the iht_size. Recall this is because
        # we need to have one set of weights for each action.
        # of shape (num actions, iht_size) here
        self.w = np.ones((self.num_actions, self.iht_size)) * self.initial_weights
        
        # We initialize self.fbtc to the version of the 
        # tile coder that we created
        self.fbtc = TileCoder(iht_size=self.iht_size, 
                                         num_tilings=self.num_tilings, 
                                         num_tiles=self.num_tiles)

    def policy(self, state, available_actions, update=False):
        """
        Implementing the epsilon greedy strategy
        :param state: the state is a dictionary that depends on the type of agent
        :param available_actions: the actions that are available so that the 
        agent stays in the grid and doesn't move towards an obstacle. 
        :return: the chosen action
        """
        action_values = []
        chosen_action = None
        # Process the state
        if not update:
            state = self.process_state(state)
        # First loop through the weights of each action and populate action_values
        # with the action value for each action and tiles instance

        # Use np.random.random to decide if an exploratory action should be taken
        # and set chosen_action to a random action if it is
        # Otherwise choose the greedy action using the given argmax 
        # function and the action values (don't use numpy's armax)
        x_seeker, y_seeker, x_hider, y_hider = state
        tiles = self.fbtc.get_tiles(x_seeker, y_seeker, x_hider, y_hider)
        action_values = [np.sum(self.w[action][tiles]) for action in range(self.num_actions)]

        if np.random.random() < self.epsilon:
            action = random.choice(available_actions)
        else:
            action = self.argmax(action_values, available_actions)
    
        return (action,action_values[action])

    def agent_start(self, state, available_actions):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state observation from the
                environment's env_start function.
        Returns:
            The first action the agent takes.
        """
        state = self.process_state(state)
        x_seeker, y_seeker, x_hider, y_hider = state
        
        # Use self.fbtc to set active_tiles using x and y
        # set current_action to the epsilon greedy chosen action using
        # the select_action function above with the active tiles
        
        # ----------------
        # COMPLETE HERE
        active_tiles =  self.fbtc.get_tiles(x_seeker, y_seeker, x_hider, y_hider) 
        current_action, _ = self.policy(state, available_actions, update=True)
        # # ----------------
        
        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
        return self.last_action

    def update(self, state, reward, available_actions):
        """
        We have already done: first state to first action, received a reward and next state for that
        now we need to take an action and update
        :param state: the state/observation returned after the last action taken
        :param action: the action chosen
        :param reward: the reward received after the last action taken
        :param terminated: has the episode terminated after taking the action in the state
        :param next_state: to what state have we transitioned?
        :return: chosen action
        """
        # choose the action here
        state = self.process_state(state, update=True)
        x_seeker, y_seeker, x_hider, y_hider = state
        
        # Use self.fbtc to set active_tiles using x and y
        # set current_action and action_value to the epsilon greedy chosen action using
        # the select_action function above with the active tiles
        
        # Update self.w at self.previous_tiles and self.previous action
        # using the reward, action_value, self.discount_factor, self.w,
        # self.step_size, and the episodic semi gradient Sarsa update from the textbook
        
        # ----------------
        active_tiles = self.fbtc.get_tiles(x_seeker, y_seeker, x_hider, y_hider)
        current_action, action_value = self.policy(state, available_actions, update=True) 
        # a real value
        update_target = reward + self.discount_factor * action_value - np.sum(self.w[self.last_action][self.previous_tiles])
        self.w[self.last_action][self.previous_tiles] +=  self.step_size * update_target * np.ones(len(active_tiles))
        # # ----------------
        
        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
        #return self.last_action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        # Update self.w at self.previous_tiles and self.previous action
        # using the reward, self.gamma, self.w,
        # self.step_size, and the Sarsa update from the textbook
        update_target = reward - np.sum(self.w[self.last_action][self.previous_tiles])
        self.w[self.last_action][self.previous_tiles] += self.step_size * update_target * np.ones(len(self.previous_tiles))
    
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon-self.epsilon_decay)