import math
import gym
from gym import spaces, logger
import numpy as np

class Cell(object):
    def __init__(self, idx, N):
        self.N = N
        self.idx = idx
        self.coordinate = np.unravel_index(action, (self.N, self.N))

class Reversi(gym.Env):
    """
    Description:
        Reversi game environment
    Observation:
        a list of (NxN) values of -1 or 0 or  (N is even)
    Actions:
        Type: Discrete(N^2)
        Represents the chosen cell to place the next token.
        The game board is then represented as flattened matrix of N^2 integers.
    Reward:
        Reward is the difference number of player's and oponent's tokens.
    Starting State:
        Empty board with just 4 tokens in the middle, arranged in diagonals:
            -------------
            | | | | | | |
            | | | | | | |
            | | |x|o| | |
            | | |o|x| | |
            | | | | | | |
            | | | | | | |
            -------------
    Episode Termination:
        Board is full of tokens
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, N=6):
        assert N % 2 == 0, "N has to be even"
        self.N = N
        self.action_space = spaces.Discrete(self.N**2)
        self.viewer = None
        self.state = None
        self.cur_player = None
        self.steps_beyond_done = None
        self.neighbours = np.array(np.meshgrid([-1,0,1],[-1,0,1])).T.reshape(-1,2)
        self.reset()

    def _is_legal_action(self, action):
        if not self.action_space.contains(action):
            return False
        if self.state.reshape(-1)[action] != 0:
            return False
        action_coordinates = np.unravel_index(action, (self.N, self.N))
        if not self._has_occupied_neighbours(action_coordinates):
            return False

        return True
        
    def _has_occupied_neighbours(self, coordinate):
        neighbours = self.neighbours + coordinate
        neighbours = neighbours[np.where(np.logical_and(neighbours[:,0]>=0, neighbours[:,0]<self.N))]
        neighbours = neighbours[np.where(np.logical_and(neighbours[:,1]>=0, neighbours[:,1]<self.N))]
        for n in neighbours:
            if self.state[n[0],n[1]] != 0:
                return True
        return False

    def step(self, action):
        assert self._is_legal_action(action), "Illegal action"
        action_coordinates = np.unravel_index(action, (self.N, self.N))
        self.state.reshape(-1)[action] = self.cur_player
        self.cur_player *= -1


        done = bool((self.state != 0).all())

        if not done:
            sum1 = (self.state == -1).sum()
            sum2 = (self.state == 1).sum()
            reward = sum2 - sum1
        elif sum2 > sum1:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1000.0
        else:
            reward = -1000.0

        self.render()
        return np.asanyarray(self.state), reward, done, {}

    def reset(self):
        self.state = np.zeros((self.N,self.N), dtype=int)
        p = int(self.N/2)
        self.state[p-1][p-1] = 1
        self.state[p][p] = 1
        self.state[p-1][p] = -1
        self.state[p][p-1] = -1
        self.steps_beyond_done = None
        self.cur_player = 1 if np.random.choice(2) == 1 else -1

    def render(self, mode='human'):
        symbols = ["x", " ", "o"]
        render = []
        columns = " "*3 + " ".join([str(i+1) for i in range(self.N)])
        render.append(columns)
        separator = " "*2 + "-"*(2*self.N + 1)
        render.append(separator)

        for row in range(self.N):
            line = chr(65+row) + " |"
            line += "|".join([symbols[c+1] for c in self.state[row]]) + "|"
            render.append(line)
        
        render.append(separator)

        if mode == 'human':
            for line in render:
                print(line)

        if mode == 'ansi':
            return "\n".join(render)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


game = Reversi(6)
game.step(9)
game.state = np.random.choice(3,36).reshape(-1,6).astype(int) -1
game.render()