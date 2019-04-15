"""
    FrozenLake problem adapted from gym with customized maps and more methods
    https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
    
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following
        SFFF
        FHFH
        FFFH
        HFFG
    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located
    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    
    """    

import sys
from contextlib import closing

import numpy as np
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "5x5" : [
        "FFFFS",
        "FHHHH",
        "FFFFF",
        "HHHHF",
        "GFFFF"
    ],
    
    "10x10" : [
        "FFFFFFFFFS",
        "FFFFFFFFFF",
        "FFHHHHHHHH",
        "FFHHHHHHHH",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "HHHHHHHHFF",
        "HHHHHHHHFF",
        "FFFFFFFFFF",
        "GFFFFFFFFF"
    ],
    
    "15x15" : [
        "FFFFFFFFFFFFFFS",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFHHHHHHHHHHHH",
        "FFFHHHHHHHHHHHH",
        "FFFHHHHHHHHHHHH",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "HHHHHHHHHHHHFFF",
        "HHHHHHHHHHHHFFF",
        "HHHHHHHHHHHHFFF",
        "GGGFFFFFFFFFFFF",
        "GGGFFFFFFFFFFFF",
        "GGGFFFFFFFFFFFF"
    ],

   
    "20x20" : [
        "SFFFFHHHHHFFFFFHHHFF",
        "FFFFFFHHHFFFFFFHHHFF",
        "FFFHFFFHFFFHHFFFFFFF",
        "FFFFFFFFFFHHHHFFHHFF",
        "FFFFHHFFFFFHHFFFHHFF",
        "FFFFHHFFFFFFFFFHHHFF",
        "FFFFFFFFHHHHFFFHHHFF",
        "HFFFFHFFFFHHFFFHHHFF",
        "HHFFFHFFFFFFFFHHHHFF",
        "HHHFFHFFFFFFFFHHHHFF",
        "HHHHFFHFFFFFHHHHHHFF",
        "HHHHFFHFFFFFFFHHHHFF",
        "HHHFFFFFFHHHFFHHHHFF",
        "HHFFFFFFFHHHFFFHHHFF",
        "HFFFFHHFFFFFHFFHHHFF",
        "FFFFHHHHFFFFFFFHHHFF",
        "FFFFFHHFFFFFFFFFFFFF",
        "FHHFFFFFFFFFFFFFFHFF",
        "HHHHFFFFFFFFFFFFFFFF",
        "HHHHFFFFFHHFFFHHHHFG"
    ]
}

           
def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # BFS to check that it's a valid path.
    def is_valid(arr, r=0, c=0):
        if arr[r][c] == 'G':
            return True

        tmp = arr[r][c]
        arr[r][c] = "#"

        # Recursively check in all four directions.
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for x, y in directions:
            r_new = r + x
            c_new = c + y
            if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                continue

            if arr[r_new][c_new] not in '#H':
                if is_valid(arr, r_new, c_new):
                    arr[r][c] = tmp
                    return True

        arr[r][c] = tmp
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]


class FrozenLakeEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4",is_slippery=True, hole_reward = False):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        #self.reward_range = (0, 1)
        self.hole_reward = hole_reward
        self.hole_reward_val = -0.1

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
               
        def to_s(row, col):
            return row*ncol + col
        
        def inc(row, col, a):
            if a == LEFT:
                col = max(col-1,0)
            elif a == DOWN:
                row = min(row+1,nrow-1)
            elif a == RIGHT:
                col = min(col+1,ncol-1)
            elif a == UP:
                row = max(row-1,0)
            return (row, col)
        

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a-1)%4, a, (a+1)%4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                rew = float(newletter == b'G')
                                if self.hole_reward and newletter == b'H':
                                    rew = self.hole_reward_val
                                # 0.8 probability to take the target action, 0.1 probability to take each neighbor action
                                if b == a:
                                    li.append((0.8, newstate, rew, done))
                                else:
                                    li.append((0.1, newstate, rew, done))
                                
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            rew = float(newletter == b'G')
                            if (self.hole_reward) and (newletter == b'H'):
                                rew = self.hole_reward_val
                            li.append((1.0, newstate, rew, done))
                           
                                                                              

        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
  
    def colors(self):
        return {
            b'S': 'green',
            b'F': 'skyblue',
            b'H': 'grey',
            b'G': 'gold',
        }

    def directions(self):
        return {
            3: '⬆',
            2: '➡',
            1: '⬇',
            0: '⬅'
        }