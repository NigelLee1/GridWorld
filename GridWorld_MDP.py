# -*- coding: utf-8 -*-
"""
COSC-4117EL: Assignment 2 Problem Domain

This code provides a basic and interactive grid world environment where a robot can navigate using the arrow keys. The robot encounters walls that block movement, gold that gives positive rewards, and traps that give negative rewards. The game ends when the robot reaches its goal. The robot's score reflects the rewards it collects and penalties it incurs.

"""

import pygame
import numpy as np
import random
from collections import defaultdict
import time
import threading as th

# Constants for our display
GRID_SIZE = 10  # Easily change this value
CELL_SIZE = 60  # Adjust this based on your display preferences
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE
GOLD_REWARD = 10
TRAP_PENALTY = -10
# GOLD_REWARD = 5
# TRAP_PENALTY = -10

ROBOT_COLOR = (0, 128, 255)
GOAL_COLOR = (0, 255, 0)
WALL_COLOR = (0, 0, 0)
EMPTY_COLOR = (255, 255, 255)
GOLD_COLOR = (255, 255, 0)  # Yellow
TRAP_COLOR = (255, 0, 0)  # Red

NOISE = 0.2
DISCOUNT = 0.9
# DISCOUNT = 0.3
# DISCOUNT = 0.6
GOAL_REWARD = 200
# GOAL_REWARD = 100
# GOAL_REWARD = 50

random.seed(100)


class GridWorld:
    UP = (-1, 0)
    DOWN = (+1, 0)
    RIGHT = (0, +1)
    LEFT = (0, -1)
    KEYS_ACTIONS = {UP: pygame.K_UP, DOWN: pygame.K_DOWN, RIGHT: pygame.K_RIGHT, LEFT: pygame.K_LEFT}

    EXPECTIMAX_UP = {UP: 1 - NOISE, RIGHT: NOISE / 3, LEFT: NOISE / 3, DOWN: NOISE / 3 }
    EXPECTIMAX_DOWN = {DOWN: 1 - NOISE, RIGHT: NOISE / 3, LEFT: NOISE / 3, UP: NOISE / 3}
    EXPECTIMAX_RIGHT = {RIGHT: 1 - NOISE, UP: NOISE / 3, DOWN: NOISE / 3, LEFT: NOISE / 3}
    EXPECTIMAX_LEFT = {LEFT: 1 - NOISE, UP: NOISE / 3, DOWN: NOISE / 3, RIGHT: NOISE / 3}

    ACTIONS = {UP: EXPECTIMAX_UP, DOWN: EXPECTIMAX_DOWN,
               RIGHT: EXPECTIMAX_RIGHT, LEFT: EXPECTIMAX_LEFT}

    def __init__(self, size=GRID_SIZE):
        self.values = None
        self.q_values = None
        self.optimal_policy = None
        self.size = size
        self.grid = np.zeros((size, size))
        # Randomly select start and goal positions
        self.start = (random.randint(0, size - 1), random.randint(0, size - 1))
        self.goal = (random.randint(0, size - 1), random.randint(0, size - 1))
        self.robot_pos = self.start
        self.score = 0
        self.step_taken = 0
        self.generate_walls_traps_gold()

    def generate_walls_traps_gold(self):

        for i in range(self.size):
            for j in range(self.size):
                if (i, j) != self.start and (i, j) != self.goal:
                    rand_num = random.random()
                    if rand_num < 0.1:  # 10% chance for a wall
                        self.grid[i][j] = np.inf
                    elif rand_num < 0.2:  # 20% chance for gold
                        self.grid[i][j] = GOLD_REWARD
                    elif rand_num < 0.3:  # 30% chance for a trap
                        self.grid[i][j] = TRAP_PENALTY
        self.grid[self.goal[0]][self.goal[1]] = GOAL_REWARD

    def move(self, direction):
        """Move the robot in a given direction."""
        x, y = self.robot_pos
        # Conditions check for boundaries and walls
        if direction == "up" and x > 0 and self.grid[x - 1][y] != np.inf:
            x -= 1
        elif direction == "down" and x < self.size - 1 and self.grid[x + 1][y] != np.inf:
            x += 1
        elif direction == "left" and y > 0 and self.grid[x][y - 1] != np.inf:
            y -= 1
        elif direction == "right" and y < self.size - 1 and self.grid[x][y + 1] != np.inf:
            y += 1
        reward = self.grid[x][y] * (DISCOUNT ** self.step_taken) - 1 # step penalty
        self.step_taken = self.step_taken + 1
        self.robot_pos = (x, y)
        self.grid[x][y] = 0  # Clear the cell after the robot moves
        self.score += reward
        return reward

    def display(self):
        """Print a text-based representation of the grid world (useful for debugging)."""
        for i in range(self.size):
            row = ''
            for j in range(self.size):
                if (i, j) == self.robot_pos:
                    row += 'R '
                elif self.grid[i][j] == np.inf:
                    row += '# '
                else:
                    row += '. '
            print(row)

    def get_states(self):
        return [(i, j) for i in range(self.size)
                    for j in range(self.size) if self.grid[i][j] != np.inf]

    def get_transition_states_and_probs(self, state, action):
        if state == self.goal:
            return [(self.goal, GOAL_REWARD)]

        result = []
        expectimax = self.ACTIONS.get(action)
        for transition in expectimax:
            row = state[0] + transition[0]
            col = state[1] + transition[1]
            s_prime = (row if 0 <= row < self.size else state[0],
                        col if 0 <= col < self.size else state[1])
            if self.grid[s_prime[0]][s_prime[1]] == np.inf:
                s_prime = state
            prob = expectimax[transition]
            result.append((s_prime, prob))
        return result

    def get_reward(self, state):
        reward = self.grid[state[0]][state[1]] - 1
        return reward

    def is_terminal(self, state):
        return state == self.goal

    def get_qvalue_from_values(self, state, action, values):
        v = 0
        for state_prime, probability in self.get_transition_states_and_probs(state, action):
            v += probability * (self.get_reward(state) + DISCOUNT * values[state_prime])
        return v

    def iterate_values_converage(self):
        values = defaultdict(lambda: 0)
        reach_threshold = 0;
        iterations = 0;
        while reach_threshold == 0:
            vnext = defaultdict(lambda: 0)
            for state in self.get_states():
                if not self.is_terminal(state):
                    maximum = float("-inf")
                    for action in [self.UP, self.DOWN, self.RIGHT, self.LEFT]:
                        qvalue = self.get_qvalue_from_values(state, action, values)
                        maximum = max(maximum, qvalue)
                    vnext[state] = maximum
                else:
                    vnext[state] = GOAL_REWARD
            iterations = iterations + 1
            max_diff = 0
            for key, value in vnext.items():
                diff = value - values[key];
                if(diff > max_diff):
                    max_diff = diff
            if (max_diff < 0.0001):
                reach_threshold = 1
            values = vnext
        print(f"it took : {iterations} iterations to figure out an optimal policy, " +
              "the convergence condition is changes between episodes are below " +
              "a small threshold, e.g. 0.0001")
        self.values = values

    def compute_qvalues(self):
        qvalues = {}
        for state in self.get_states():
            if not self.is_terminal(state):
                for action in [self.UP, self.DOWN, self.RIGHT, self.LEFT]:
                    qvalues[state, action] = self.get_qvalue_from_values(state, action, self.values)
                    qvalues[state, action]
        self.q_values = qvalues

    def compute_optimal_policy(self):
        policy = {}
        for state in self.get_states():
            if not self.is_terminal(state):
                maximum = -float("inf")
                for action in self.ACTIONS.keys():
                    qvalue = self.get_qvalue_from_values(state, action, self.values)
                    if qvalue > maximum:
                        maximum = qvalue
                        bestact = action
                policy[state] = bestact
        self.optimal_policy = policy

        is_terminal = False
        path = {}
        agent_pos = self.robot_pos
        while is_terminal == False:
            action = self.optimal_policy[agent_pos]
            if path.get(agent_pos) is not None:
                break;
            path[agent_pos] = action
            row = agent_pos[0] + action[0]
            col = agent_pos[1] + action[1]
            agent_pos = (row, col)
            if row < 0 or row >= self.size or col < 0 or col >= self.size \
                    or self.grid[row][col] == np.inf or (row, col) == self.goal:
                is_terminal = True

        self.path = path



def setup_pygame():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Grid World")
    clock = pygame.time.Clock()
    return screen, clock


def draw_grid(world, screen):
    GRAY = (200, 200, 200)
    font2 = pygame.font.SysFont('didot.ttc', 30)
    actions = {world.UP: font2.render('^', True, GRAY), world.RIGHT: font2.render('>', True, GRAY),
              world.DOWN: font2.render('v', True, GRAY), world.LEFT: font2.render('<', True, GRAY)}

    Green = (0, 255, 0)
    path_font = pygame.font.SysFont('didot.ttc', 40)
    path_actions = {world.UP: path_font.render('^', True, Green), world.RIGHT: path_font.render('>', True, Green),
               world.DOWN: path_font.render('v', True, Green), world.LEFT: path_font.render('<', True, Green)}

    """Render the grid, robot, and goal on the screen."""
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            # Determine cell color based on its value
            color = EMPTY_COLOR
            cell_value = world.grid[i][j]
            if cell_value == np.inf:
                color = WALL_COLOR
            elif cell_value == GOLD_REWARD:  # Gold
                color = GOLD_COLOR
            elif cell_value == TRAP_PENALTY:  # Trap
                color = TRAP_COLOR
            pygame.draw.rect(screen, color, pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Drawing the grid lines
    for i in range(GRID_SIZE):
        pygame.draw.line(screen, (200, 200, 200), (i * CELL_SIZE, 0), (i * CELL_SIZE, SCREEN_HEIGHT))
        pygame.draw.line(screen, (200, 200, 200), (0, i * CELL_SIZE), (SCREEN_WIDTH, i * CELL_SIZE))

    pygame.draw.circle(screen, ROBOT_COLOR,
                       (int((world.robot_pos[1] + 0.5) * CELL_SIZE), int((world.robot_pos[0] + 0.5) * CELL_SIZE)),
                       int(CELL_SIZE / 3))

    pygame.draw.circle(screen, GOAL_COLOR,
                       (int((world.goal[1] + 0.5) * CELL_SIZE), int((world.goal[0] + 0.5) * CELL_SIZE)),
                       int(CELL_SIZE / 3))

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if world.grid[i][j] != np.inf and (i,j) != world.goal:
                if (i, j) in world.path.keys():
                    screen.blit(path_actions[world.path[(i, j)]], (j * CELL_SIZE + 20, i * CELL_SIZE + 20))
                else:
                    screen.blit(actions[world.optimal_policy[(i, j)]], (j * CELL_SIZE + 20, i * CELL_SIZE + 20))

AUTO = False
def star_automation():
    global AUTO
    AUTO = True

def main():
    """Main loop"""
    screen, clock = setup_pygame()
    world = GridWorld()
    # get the start time
    st = time.time()
    world.iterate_values_converage()
    world.compute_qvalues()
    world.compute_optimal_policy()
    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    print(f'the value (utility) of the starting point: {world.values[(world.start)]}')

    S = th.Timer(1, star_automation)
    S.start()

    running = True
    while running:

        if AUTO:
            best_action = world.KEYS_ACTIONS[world.path[world.robot_pos]]
            key_event = pygame.event.Event(pygame.KEYDOWN, {"unicode":123,"key":best_action,"mod":pygame.KMOD_ALT})
            pygame.event.post(key_event)
            pygame.time.wait(1000)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # Move robot based on arrow key press
                if event.key == pygame.K_UP:
                    world.move("up")
                if event.key == pygame.K_DOWN:
                    world.move("down")
                if event.key == pygame.K_LEFT:
                    world.move("left")
                if event.key == pygame.K_RIGHT:
                    world.move("right")
                # Print the score after the move
                print(f"Current Score: {world.score}")
                # Check if the robot reached the goal
                if world.robot_pos == world.goal:
                    print("Robot reached the goal!")
                    print(f"Final Score: {world.score}")
                    running = False
                    break
        # Rendering
        screen.fill(EMPTY_COLOR)
        draw_grid(world, screen)
        pygame.display.flip()

        clock.tick(10)  # FPS

    pygame.quit()


if __name__ == "__main__":
    main()