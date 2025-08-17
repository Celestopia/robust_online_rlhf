# file: constants.py
GRID_SIZE = 11
assert GRID_SIZE % 2 == 1, "Grid size must be odd."
STATE_DIM = GRID_SIZE ** 2
ACTION_DIM = 5
TRAJECTORY_LENGTH = 5
REWARD_VEC = [0.5, 0.8, 0.7, 0.3, 0.2]
EPSILON_0 = 0.01
DELTA = 0.01