import numpy as np
import pygame as pg
import random
##########################Setup Environment <3##########################

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
        


        