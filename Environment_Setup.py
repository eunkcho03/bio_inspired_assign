import numpy as np
import pygame as pg
import random
##########################Setup Environment <3##########################
# I am going to build a grid world environment with traps, portals, and sliding blocks and also decoys. The agent will start at a given position and must reach the goal while avoiding traps and sliding blocks. The agent can also use portals to teleport to other locations in the grid. The agent will receive a reward for reaching the goal, a penalty for falling into a trap, and a small penalty for each step taken. The agent will also receive a small penalty for using a portal. The agent will also receive a small reward  for finding a decoy, but there is a probability that the decoy is fake and will result in a penalty instead. The agent will have a slip probability that will cause it to move in a random direction instead of the intended direction. The environment will be represented as a 2D grid with different symbols representing the different elements of the environment.

class Environment:
    def __init__(self, cfg):
        # integers!!
        self.H = int(cfg['grid_size'])
        self.W = int(cfg['grid_size'])
        self.n_slide_block = int(cfg['n_slide_block'])
        self.n_trap = int(cfg['n_trap'])
        self.max_step = int(cfg['max_step'])

        self.n_portal = int(cfg['n_portal']) # number of portal must be even!!
        self.n_decoy = int(cfg['n_decoy'])

        # floats!!
        self.step_cost = float(cfg['step_cost'])
        self.goal_reward = float(cfg['goal_reward'])
        self.trap_cost = float(cfg['trap_cost'])
        self.portal_cost = float(cfg['portal_cost'])
        self.slip_prob = float(cfg['slip_prob'])
        self.decoy_reward = float(cfg['decoy_reward'])
        self.decoy_cost = float(cfg['decoy_cost'])
        self.decoy_prob_reward = float(cfg['decoy_prob_reward'])
        
        
        self.start = cfg['start']
        self.goal = cfg['goal']
        