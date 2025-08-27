import numpy as np
import pygame as pg
import random
from __future__ import annotations
from typing import List, Tuple, Optional 
from dataclasses import dataclass

##########################Setup Environment <3##########################
up, right, down, left = 0, 1, 2, 3
actions = [up, right, down, left]
directions = {up: (0, -1), right: (1, 0), down: (0, 1), left: (-1, 0)}

T_emp = 0
T_wall = 1
T_trap = 2
t_slide = 3
T_decoy = 4
T_portal = 5
T_depot = 6
T_pickup = 7

class Environment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = random.Random(cfg.get("seed"))
        self.np_rng = np.random.default_rng(cfg.get("seed"))
        self.agent_start = self.depot_pos
        self.reset()
        
    def size(self, sizing):
        H = len(sizing)
        W = len(sizing[0])
        return H,W 
    
    
    
    def ascii(self, ascii_map): # the input is string
        lines = []
        for l in ascii_map.strip().split("\n"):
            lines.append(l)
        H, W = self.size(lines)
        self.grid = []
        for _ in range(H):
            self.grid.append([T_emp] * W)
        self.picks_all=set() # empty python set
        for r in range(H):
            for c in range(W):
                ch = lines[r][c]
                if ch == 'X':
                    self.grid[r][c] = T_wall
                elif ch == 'T':
                    self.grid[r][c] = T_trap
                elif ch == 'W':
                    self.grid[r][c] = t_slide
                elif ch == 'D':
                    self.grid[r][c] = T_decoy
                elif ch == 'H':
                    self.grid[r][c] = T_depot
                elif ch == 'P':
                    self.grid[r][c] = T_portal
                elif ch.isdigit():
                    self.grid[r][c] = T_pickup
                    self.picks_all.add((r,c))
        return self.grid, self.picks_all
                
    def observation_tensor(self):
        layers = []
        H, W = self.size(self.grid)
        
        # Agent layer
        agent_layer = np.zeros((H, W), dtype=np.float32)
        agent_layer[self.agent_pos] = 1.0
        layers.append(agent_layer)
        
        # Depot layer
        depot_layer = np.zeros((H, W), dtype=np.float32)
        depot_layer[self.depot_pos] = 1.0
        layers.append(depot_layer)
        
        # Pickup Layer
        pickup_layer = np.zeros((H, W), dtype=np.float32)
        for p in self.picks_remaining:
            pickup_layer[p] = 1.0
        layers.append(pickup_layer)
        
        Tiles = [T_trap, t_slide, T_decoy, T_portal]
        
        for tile in Tiles:
            tile_layer = (np.array(self.grid) == tile).astype(np.float32)
            layers.append(tile_layer)
        
        observation = np.stack(layers, axis=0)
        return observation


        