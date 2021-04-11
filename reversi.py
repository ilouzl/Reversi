import math
import gym
from gym import spaces, logger
import numpy as np
from keyboard import get_key

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
        assert N >= 4, "N has to be larger than 3"
        self.N = N
        self.action_space = spaces.Discrete(self.N**2)
        self.viewer = None
        self.state = None
        self.cur_player = None
        self.steps_beyond_done = None
        self.board_symbols = {-1: "x", 0:" ", 1:"o", 2:"?"}
        self.neighbours = np.array(np.meshgrid([-1,0,1],[-1,0,1])).T.reshape(-1,2)
        self.neighbours = self.neighbours[(self.neighbours != 0).any(axis=1)]
        self.reset()

    def _is_in_board(self, coordinate):
        return all([0 <= v <= (self.N-1) for v in coordinate])

    def _is_legal_action(self, action):
        if not self.action_space.contains(action):
            return False
        if self.state.reshape(-1)[action] != 0:
            return False
        action_coordinates = self._idx2coordinate(action)
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
        if not self._is_legal_action(action):
            print("Illegal action")
            return False
        action_coordinates = self._idx2coordinate(action)
        self.state.reshape(-1)[action] = self.cur_player
        
        # flip oponent pieces
        opponent = self.cur_player * -1
        accumulated_flip_coords = []
        for d in self.neighbours:
            coord = np.array(action_coordinates)
            flip_coords = []
            potential_flip = False
            verified_flip = False
            while True:
                coord = tuple(np.asanyarray(coord) + d)
                if self._is_in_board(coord):
                    cell_value = self.state[tuple(coord)]
                    if not potential_flip:
                        if self.state[coord] == opponent:
                            potential_flip = True
                            flip_coords.append(coord)
                        else:
                            break     
                    else:
                        if self.state[coord] == opponent:
                            flip_coords.append(coord)
                        elif self.state[coord] == self.cur_player:
                            verified_flip = True
                            break
                        else:
                            break
                else:
                    break
            if verified_flip:
                accumulated_flip_coords.extend(flip_coords)
        
        for c in accumulated_flip_coords:
            self.state[c] = self.cur_player


        
        
        self.cur_player *= -1
        done = bool((self.state != 0).all())

        sum1 = (self.state == -1).sum()
        sum2 = (self.state == 1).sum()
        if not done:
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

    def _coordinate2idx(self, coordinate):
        return np.ravel_multi_index(coordinate, (self.N,self.N))

    def _idx2coordinate(self, idx):
        return np.unravel_index(idx, (self.N, self.N))

    def play(self, interactive=False):
        if interactive:
            coord = tuple(np.argwhere(self.state == 0)[0])
            d = [0,0]
            while True:
                new_coord = tuple(np.asanyarray(coord) + d)
                if self._is_in_board(new_coord):
                    if not self.state[new_coord] == 0:
                        coord = new_coord
                        continue
                    else:
                        coord = new_coord
                        st = np.array(self.state)
                        st[coord] = 2
                print("It's %s turn. What's your next move? [use arrow keys] : "%(self.board_symbols[self.cur_player]))
                self.render(state = st)
                k = get_key()
                if k == "UP":
                    d = [-1,0]
                elif k == "DOWN":
                    d = [1,0]
                elif k == "LEFT":
                    d = [0,-1]
                elif k == "RIGHT":
                    d = [0,1]
                elif k == "ENTER":
                    break

            row, col = coord
        else:                
            uinput = input("It's %s turn. What's your next move? [like 'B3'] : "%(self.board_symbols[self.cur_player]))
            try:
                row = ord(uinput[0].upper()) - ord('A')
                col = int(uinput[1:]) - 1
                assert self._is_in_board((row, col)), "Coordinate outside the board"
            except Exception as e:
                print(e)
                return
        action = self._coordinate2idx((row,col))
        return self.step(action)

    def render(self, mode='human', state=None):
        if state is None:
            state = self.state
        render = []
        columns = " "*3 + " ".join([str(i+1) for i in range(self.N)])
        render.append(columns)
        separator = " "*2 + "-"*(2*self.N + 1)
        render.append(separator)

        for row in range(self.N):
            line = chr(65+row) + " |"
            line += "|".join([self.board_symbols[c] for c in state[row]]) + "|"
            render.append(line)
        
        render.append(separator)

        if mode == 'human':
            for line in render:
                print(line)

        if mode == 'ansi':
            return "\n".join(render)


game = Reversi(4)
game.render()
done = False
while True:
    state, reward, done, _ = game.play(interactive=True)
    print("Reward is %d"%(reward))
    if done: break