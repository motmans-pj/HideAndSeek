from pettingzoo.utils.env import ParallelEnv
import functools
import random
import pygame
from copy import copy
import numpy as np
import gymnasium as gym
from pettingzoo.test import parallel_api_test
from collections import defaultdict
from matplotlib import pyplot as plt


class HideAndSeek(ParallelEnv):
    # Some data that stays constant for the whole environment
    metadata = {'render_modes': ['human'], "render_fps": 4}

    def __init__(self, size, sight_len, n_obstacles=2, render_mode="human", static=False):
        '''
        Constructor for the HideAndSeek environment
        This function initializes the attributes for this environment: the action spaces,
        observation spaces, possible agents and render mode.


        Size: the size of the gridworld
        Sight Length: the number of grids spanned by the line of sight
        Number of obstacles: the number of obstacles
        Static: if True, the hider can not move => stage 1 of the game
        '''

        self.possible_agents = ["Seeker", "Hider"]
        self.timestep = 0
        self.n_obstacles = n_obstacles
        self.render_mode = render_mode
        self.size = size
        self.sight_len = sight_len
        # For the line of sight, we need an orientation of the seeker
        # if we keep it at 4 actions, the orientation will always be clear from the movement
        self.seeker_orientation = None
        # the line of sight: will become an array of locations
        self.seeker_line_of_sight = []
        self.static = static

        # The agents can do one of four actions: move up, right, left, down
        # We said first that we would consider as well just turning, but
        # this might be a bit more complicated when trying to keep track of
        # where the agent is looking and where the line of sight should be.
        self.action_spaces = {agent: gym.spaces.Discrete(4) for agent in self.possible_agents}

        # Draws from the tutorial: have to check that the gym.spaces.Box implementation is correct
        # It looks like for the observation spaces, we have to define the observations for the seeker and hider separately
        # The seeker observes the obstacles location, its own location, the location of the 'hider' and its orientation
        # The hider observes its location and the obstacles

        self.observation_spaces = gym.spaces.Dict(
            {"Seeker": gym.spaces.Dict({"Seeker_location": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                                        "Hider_location": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                                        "Obstacles": gym.spaces.Tuple(
                                            [gym.spaces.Box(0, size - 1, shape=(2,), dtype=int) for _ in
                                             range(n_obstacles)])
                                        # "Orientation": gym.spaces.Discrete(4)
                                        }),
             "Hider": gym.spaces.Dict({"Seeker_location": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                                       "Hider_location": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                                       "Obstacles": gym.spaces.Tuple(
                                           [gym.spaces.Box(0, size - 1, shape=(2,), dtype=int) for _ in
                                            range(n_obstacles)])
                                       })
             })

        # This would lead to an enormous amount of states

        # self.observation_spaces = gym.spaces.Dict(
        #     {"Seeker": gym.spaces.Dict({"Seeker_location":gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #                                "Obstacles": gym.spaces.Tuple([gym.spaces.Box(0, size - 1, shape=(2,), dtype=int) for _ in range(n_obstacles)]),
        #                                "LineOfSight":gym.spaces.Tuple([gym.spaces.Box(0, size - 1, shape=(2,), dtype=int) for _ in range(size - 1)])
        #                                 }),
        #      "Hider": gym.spaces.Dict({"Hider_location":gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #                                "Obstacles":gym.spaces.Tuple([gym.spaces.Box(0, size - 1, shape=(2,), dtype=int) for _ in range(n_obstacles)])
        #      })
        # })

        # Making a lookup table for actions (don't know if we'll need it)
        self.action_space_lut = {0: 'Down', 1: 'Right', 2: 'Up', 3: 'Left'}

        # Mapping actions to directions

        self._action_to_direction = {
            0: np.array([1, 0]),  # Down
            1: np.array([0, 1]),  # Right
            2: np.array([-1, 0]),  # Up
            3: np.array([0, -1]),  # Left
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, return_info=False, options=None):
        '''
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        '''

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        # termination if found
        self.terminations = {agent: False for agent in self.agents}
        # truncation if time limit reached f.e.
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.timestep = 0

        # Initializing the locations of hider, seeker and obstacles: similar to tutorial

        locations = []
        #  + self.n_obstacles (initialize more locations if needed)
        while len(locations) < 2:
            # only randomly generate two locations, obstacles fixed
            loc = list(np.random.randint(low=0, high=self.size, size=2, dtype=int))
            if loc not in locations:
                locations.append(loc)
        # In the static case, fix the hider at the same place
        if self.static:
            self.hider_location = [self.size - 1, self.size - 1]
            self.seeker_location = [0, 0]
        else:
            self.hider_location = locations[0]
            self.seeker_location = locations[1]

        # Here we choose to have three obstacles at fixed locations
        self._is_obstacle = tuple([[2, 2], [self.size - 2, self.size - 2], ])
        # self._is_obstacle = tuple(list(loc) for loc in locations[2:])
        # In the beginning, we want no line of sight, so we set it equal to the seeker location
        self.line_of_sight = tuple([])

        # Using this, we can probably make a line of sight from the beginning
        # start the seeker by orienting downwards
        self.seeker_orientation = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation

    def _get_obs(self):
        '''
        Function that returns the observation, it should follow the same structure
        as the observation space
        '''
        return {
            "Seeker": {"Seeker_location": self.seeker_location,
                       "Hider_location": self.hider_location,
                       "Obstacles": self._is_obstacle
                       # "Orientation": self.seeker_orientation
                       },
            "Hider": {"Seeker_location": self.seeker_location,
                      "Hider_location": self.hider_location,
                      "Obstacles": self._is_obstacle}
        }

    def _get_info(self):
        '''
        Returns some information, similar to the tutorial, one piece
        of information we can return is the distance between hider and seeker
        Must give info to all live agents
        '''

        return {
            "Seeker": np.linalg.norm(np.array(self.seeker_location) - np.array(self.hider_location)),
            "Hider": np.linalg.norm(np.array(self.seeker_location) - np.array(self.hider_location))
        }

    def select_random_actions(self):

        '''
        Function to select a random action among the available ones
        '''

        available_actions_seeker = []
        available_actions_hider = []
        obstacle_locations = [list(l) for l in self._is_obstacle]

        # restricting the actions that would direct to a non obstacle cell within the
        # field
        for i in range(4):
            # loop over every possible action
            temp_seeker = list(self.seeker_location + self._action_to_direction[i])
            if temp_seeker not in obstacle_locations:
                if (temp_seeker[0] <= self.size - 1) and (temp_seeker[1] <= self.size - 1) and (
                        temp_seeker[0] >= 0) and (temp_seeker[1] >= 0):
                    available_actions_seeker.append(i)

        for j in range(4):
            temp_hider = list(self.hider_location + self._action_to_direction[j])
            if temp_hider not in obstacle_locations:
                if (temp_hider[0] <= self.size - 1) and (temp_hider[1] <= self.size - 1) and (temp_hider[0] >= 0) and (
                        temp_hider[1] >= 0):
                    available_actions_hider.append(j)

        self.actions = {"Seeker": random.choice(available_actions_seeker),
                        "Hider": random.choice(available_actions_hider)}

    def f_available_actions(self):

        '''
        Returns the available actions
        '''

        available_actions_seeker = []
        available_actions_hider = []
        obstacle_locations = [list(l) for l in self._is_obstacle]

        # restricting the actions that would direct to a non obstacle cell within the
        # field
        for i in range(4):
            # loop over every possible action
            temp_seeker = list(self.seeker_location + self._action_to_direction[i])
            if temp_seeker not in obstacle_locations:
                if (temp_seeker[0] <= self.size - 1) and (temp_seeker[1] <= self.size - 1) and (
                        temp_seeker[0] >= 0) and (temp_seeker[1] >= 0):
                    available_actions_seeker.append(i)

        for j in range(4):
            temp_hider = list(self.hider_location + self._action_to_direction[j])
            if temp_hider not in obstacle_locations:
                if (temp_hider[0] <= self.size - 1) and (temp_hider[1] <= self.size - 1) and (temp_hider[0] >= 0) and (
                        temp_hider[1] >= 0):
                    available_actions_hider.append(j)

        self.available_actions = {"Seeker": available_actions_seeker,
                                  "Hider": available_actions_hider}
        return self.available_actions

    def step(self, actions):
        '''
        Implements a step
        actions: a dictionary with the action for each agent
        '''

        found = False
        obstacle_locations = [list(l) for l in self._is_obstacle]

        # computing the direction of the taken action

        direction_seeker = self._action_to_direction[self.actions["Seeker"]]
        self.seeker_orientation = self.actions["Seeker"]
        direction_hider = self._action_to_direction[self.actions["Hider"]]

        self.seeker_location = list(self.seeker_location + direction_seeker)

        if not self.static:
            # If static, do not change location
            self.hider_location = list(self.hider_location + direction_hider)

        # resetting the lenght of the line of sight
        self.line_of_sight = list(self.seeker_location for _ in range(self.sight_len))

        # Implementing the line of sight logic
        broken = False
        for i in range(len(self.line_of_sight)):
            # has the line of sight been broken?
            next_grid = list(self.seeker_location + (i + 1) * direction_seeker)

            # if the next grid on which the line of sight would be placed is not an obstacle nor out of the grid
            if (not next_grid in obstacle_locations) and (next_grid[0] < self.size) and (next_grid[1] < self.size) and (
                    next_grid[0] >= 0) and (next_grid[1] >= 0):
                self.line_of_sight[i] = next_grid
            else:  # if out of the grid or obstacle, the line of sight breaks
                broken = True
                breaking_point = i
                break
        # if the line of sight has broken, we set the breaking point and all locations after the breaking point to that of the one before the breaking point
        if broken:
            if breaking_point == 0:
                self.line_of_sight = []
            else:
                self.line_of_sight = self.line_of_sight[:breaking_point]

        obs = self._get_obs()
        info = self._get_info()

        if (self.hider_location in self.line_of_sight) or (self.hider_location == self.seeker_location):
            self.terminations["Hider"] = True
            # If the episode is over: terminate the episode, reset and return info
            self._cumulative_rewards["Seeker"] += 100
            self._cumulative_rewards["Hider"] -= 100
            self.rewards["Seeker"] = +100
            self.rewards["Hider"] = -100
            # env.reset()
            return obs, self.rewards, self.terminations, self.truncations, info

        # Truncated: early stopping, for example after 100 time steps
        self.timestep += 1  # increase the timestep with 1 each time
        # reward for seeker upon not finding the hider is -1 each timestep
        # reward for hider is +1
        self._cumulative_rewards["Seeker"] -= 1
        self._cumulative_rewards["Hider"] += 1
        # Reward for this action in this state
        self.rewards["Seeker"] = -1
        self.rewards["Hider"] = +1
        # Truncate the episode at a certain time

        if self.timestep >= 80:
            self.truncations = {agent: True for agent in self.agents}
            # env.reset()
            return obs, self.rewards, self.terminations, self.truncations, info

        return obs, self.rewards, self.terminations, self.truncations, info

    def render_text(self, d=1):
        grid = np.zeros((self.size, self.size), dtype=int)
        grid = grid.astype(str)
        grid[grid == '0'] = '.'

        # start with line of sight because seeker should override it
        for line in self.line_of_sight:
            if (self.seeker_orientation == 0) or (self.seeker_orientation == 2):
                grid[line[0], line[1]] = '|'
            else:
                grid[line[0], line[1]] = '-'

        grid[self.seeker_location[0], self.seeker_location[1]] = 'S'
        grid[self.hider_location[0], self.hider_location[1]] = 'H'
        for obst_loc in self._is_obstacle:
            grid[obst_loc[0], obst_loc[1]] = 'X'
        print(f"{grid} \n")

    def render_plot(self):
        plt.figure(figsize=(7, 7))
        plt.xlim(0, self.size)
        plt.ylim(0, self.size)
        plt.xticks(np.arange(0, self.size, step=1))
        plt.yticks(np.arange(0, self.size, step=1))
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        obstacles = self.observation_spaces['Seeker']['Obstacles']
        for line in self.line_of_sight:
            if (self.seeker_orientation == 0) or (self.seeker_orientation == 2):
                plt.plot(line[0], line[1], 'g*', markersize=10)
            else:
                plt.plot(line[0], line[1], 'g*', markersize=10)

        for obs in self._is_obstacle:
            plt.plot(obs[1], obs[0], 'ks', markersize=20)
        seeker_location = self.observation_spaces['Seeker']['Seeker_location']
        hider_location = self.observation_spaces['Seeker']['Hider_location']
        plt.plot(self.seeker_location[0], self.seeker_location[1], 'ro', markersize=15)
        plt.plot(self.hider_location[0], self.hider_location[1], 'bo', markersize=15)
        plt.show()

    def render_rgb(self, mode="human"):
        if mode == "rgb_array":
            raise NotImplementedError()
        elif mode == "human":
            screen_size = (self.size * 50, self.size * 50)
            screen = pygame.display.set_mode(screen_size)
            pygame.display.set_caption("HideAndSeek")
            # background = pygame.Surface(screen.get_size())

            bg = pygame.image.load("bg.png")
            bg = pygame.transform.scale(bg, (500, 500))

            # INSIDE OF THE GAME LOOP
            screen.blit(bg, (-10, -10))
            # background = background.convert()
            # background.fill((255, 255, 255))

            # Load images
            seeker_image = pygame.image.load("Seeker.png").convert_alpha()
            seeker_image = pygame.transform.scale(seeker_image, (100, 100))
            hider_image = pygame.image.load("whitewalker.png").convert_alpha()
            hider_image = pygame.transform.scale(hider_image, (50, 50))
            obstacle_image = pygame.image.load("iceberg.png").convert_alpha()
            obstacle_image = pygame.transform.scale(obstacle_image, (50, 50))
            line_of_sight_image = pygame.image.load("Fire.png").convert_alpha()
            line_of_sight_image = pygame.transform.scale(line_of_sight_image, (25, 25))

            # Draw obstacles
            for obst_loc in self._is_obstacle:
                # grid[obst_loc[0], obst_loc[1]] = 'X'
                obstacle_rect = obstacle_image.get_rect()
                obstacle_rect.center = (obst_loc[0] * 50 + 25, obst_loc[1] * 50 + 25)
                screen.blit(obstacle_image, obstacle_rect)

            # Draw agents in right orientation (optional imo)

            if self.seeker_orientation == 1:
                # flip(img, flip_x,flip_y)
                seeker_image = pygame.transform.flip(seeker_image, False, False)
                seeker_rect = seeker_image.get_rect()
                seeker_rect.center = (self.seeker_location[0] * 50 + 25, self.seeker_location[1] * 50 + 25)
                screen.blit(seeker_image, seeker_rect)
            elif self.seeker_orientation == 0:
                seeker_image = pygame.transform.rotate(seeker_image, 90)
                seeker_rect = seeker_image.get_rect()
                seeker_rect.center = (self.seeker_location[0] * 50 + 25, self.seeker_location[1] * 50 + 25)
                screen.blit(seeker_image, seeker_rect)
            elif self.seeker_orientation == 3:
                seeker_image = pygame.transform.flip(seeker_image, True, True)
                seeker_rect = seeker_image.get_rect()
                seeker_rect.center = (self.seeker_location[0] * 50 + 25, self.seeker_location[1] * 50 + 25)
                screen.blit(seeker_image, seeker_rect)
            elif self.seeker_orientation == 2:
                seeker_image = pygame.transform.rotate(seeker_image, 270)
                seeker_rect = seeker_image.get_rect()
                seeker_rect.center = (self.seeker_location[0] * 50 + 25, self.seeker_location[1] * 50 + 25)
                screen.blit(seeker_image, seeker_rect)

            ##### If not in right orientation, just do this
            # seeker_rect = seeker_image.get_rect()
            # seeker_rect.center = (self.seeker_location[0] * 50 + 25, self.seeker_location[1] * 50 + 25)
            # screen.blit(seeker_image, seeker_rect)

            hider_rect = hider_image.get_rect()
            hider_rect.center = (self.hider_location[0] * 50 + 25, self.hider_location[1] * 50 + 25)
            screen.blit(hider_image, hider_rect)

            if (self.seeker_orientation == 2 or self.seeker_orientation == 0):
                line_of_sight_image = pygame.transform.rotate(line_of_sight_image, 90)

            for line in self.line_of_sight:
                line_rect = line_of_sight_image.get_rect()

                line_rect.center = (line[0] * 50 + 25, line[1] * 50 + 25)
                screen.blit(line_of_sight_image, line_rect)

            # Update the screen
            pygame.display.flip()
            pygame.time.wait(int(1000 / self.metadata["render_fps"]))
            print('--------------------------------')

            #######
            # In colab, you can not show the screen, the below is a way to deal
            # with that, I think locally you could comment everything out below
            # and uncomment the pygame.wait
            #######

            #view = pygame.surfarray.array3d(screen)

            #  convert from (width, height, channel) to (height, width, channel)
            #view = view.transpose([1, 0, 2])

            #  convert from rgb to bgr
            #img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)

            # Display image, clear cell every 0.5 seconds
            #cv2_imshow(img_bgr)
            #time.sleep(0.5)
            # output.clear()
