import sys
import os
import time
import random
import copy

import pygame
import joblib
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

seed = 100

# random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Initialize pygame
pygame.init()

class pong_env:

    def __init__(self, HEIGHT, WIDTH, PADDLE_HEIGHT, PADDLE_WIDTH, BALL_SIZE, BALL_SPEED:tuple,\
                 MISS_REWARD=-1,HIT_REWARD=10,PASSIVE_REWARD=1):
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.agent = pygame.Rect(0, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.AGENT_WIDTH = PADDLE_WIDTH
        self.AGENT_HEIGHT = PADDLE_HEIGHT
        self.ball = pygame.Rect(WIDTH // 2 - BALL_SIZE // 2, HEIGHT // 2 - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)
        self.BALL_SIZE = BALL_SIZE
        self.BALL_SPEED_X = BALL_SPEED[0]
        self.BALL_SPEED_Y = BALL_SPEED[1]

        self.__ACTION_SPACE__= {0:"up",1:"down",2:"stay"}

        self.MISS_REWARD = MISS_REWARD
        self.HIT_REWARD = HIT_REWARD
        self.PASSIVE_REWARD = PASSIVE_REWARD

        self.steps = 0
        self.hits = 0
        self.cycles = 0

    def __get_state__(self):
        return ((self.ball.right+self.ball.left)/2,(self.ball.top+self.ball.bottom)/2,(self.agent.top+self.agent.bottom)/2,
                self.BALL_SPEED_X,self.BALL_SPEED_Y)
    
    def reset(self):
        self.steps = 0
        self.hits = 0
        self.ball.right = random.randint(5, self.WIDTH)
        self.ball.left = self.ball.right - self.BALL_SIZE
        self.ball.top = random.randint(self.BALL_SIZE, self.HEIGHT)
        self.ball.bottom = self.ball.top + self.BALL_SIZE
        self.agent.top = random.randint(0, self.HEIGHT-self.AGENT_HEIGHT)
        self.agent.bottom = self.agent.top + self.AGENT_HEIGHT
        self.BALL_SPEED_X = random.choice([-7,7])
        self.BALL_SPEED_Y = random.choice([-7,7])
        return self.__get_state__()
    
    def get_hits(self):
        return self.hits
    
    def get_cycles(self):
        return self.cycles
    
    def observe(self):
        return self.__get_state__()
    
    def step(self,action):
        prev_state = self.observe()

        assert action in self.__ACTION_SPACE__.keys()
        if self.__ACTION_SPACE__[action]=="up" and self.agent.top>(0.15*self.HEIGHT):
            self.agent.y-=15
            action_index = 0
        elif self.__ACTION_SPACE__[action]=="down" and self.agent.bottom<(0.85*self.HEIGHT):
            self.agent.y+=15
            action_index = 1
        else:
            action_index = 2

        # Move ball
        self.ball.x += self.BALL_SPEED_X
        self.ball.y += self.BALL_SPEED_Y

        # Ball collision with walls
        if self.ball.top <= 0 or self.ball.bottom >= self.HEIGHT:
            self.BALL_SPEED_Y *= -1
        # Ball out of bounds
        if self.ball.right >= self.WIDTH:
            self.BALL_SPEED_X *= -1
        if self.ball.left <= 5:
            self.ball.x = random.randint(0,int(self.WIDTH*0.85)) - self.BALL_SIZE // 2
            self.ball.y = random.randint(0, self.HEIGHT) - self.BALL_SIZE // 2
            self.BALL_SPEED_X *= -1

        if self.ball.colliderect(self.agent):
            self.BALL_SPEED_X *= -1

        next_state = self.observe()

        # Compute Rewards
        if self.ball.left <= 5:
            reward = self.MISS_REWARD
            self.cycles += 1
        elif self.ball.colliderect(self.agent):
            reward = self.HIT_REWARD
            self.hits += 1
            self.cycles += 1
        else:
            diff = abs(next_state[1]-next_state[2])
            reward = (self.HEIGHT -(diff/self.HEIGHT))*0.001
            reward = self.PASSIVE_REWARD + reward



        self.steps += 1

        return prev_state, action_index, next_state, reward