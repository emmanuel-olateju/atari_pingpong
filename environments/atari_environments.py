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

# Initialize pygame
pygame.init()

# Constants
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
font = pygame.font.Font(None, 36)

class pong_env:

    def __init__(self, HEIGHT, WIDTH, PADDLE_HEIGHT, PADDLE_WIDTH, BALL_SIZE, BALL_SPEED:tuple,\
                 MISS_REWARD=-1,HIT_REWARD=10,PASSIVE_REWARD=1,RENDER_MODE=False):
        
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH

        self.agent = pygame.Rect(0, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.AGENT_WIDTH = PADDLE_WIDTH
        self.AGENT_HEIGHT = PADDLE_HEIGHT
        self.ball = pygame.Rect(WIDTH // 2 - BALL_SIZE // 2, HEIGHT // 2 - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)
        self.BALL_SIZE = BALL_SIZE
        self.BALL_SPEED_X = BALL_SPEED[0]
        self.BALL_SPEED_Y = BALL_SPEED[1]
        
        self.RENDER_MODE = RENDER_MODE
        self.init_render()

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
        self.cycles = 0
        self.ball.right = random.randint(int(0.5*self.WIDTH), int(0.9*self.WIDTH))
        self.ball.left = self.ball.right - self.BALL_SIZE
        self.ball.top = random.randint(int(0.5*self.HEIGHT), int(0.9*self.HEIGHT))
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


        """
            AGENT TAKE ACTION
        """
        # Select action if it satisfies constraints
        assert action in self.__ACTION_SPACE__.keys()
        if self.__ACTION_SPACE__[action]=="up" and self.agent.top>int(0.05*self.HEIGHT):
            self.agent.y-=15
            action_index = 0
        elif self.__ACTION_SPACE__[action]=="down" and self.agent.bottom<=int(0.95*self.HEIGHT):
            self.agent.y+=15
            action_index = 1
        elif self.__ACTION_SPACE__[action]=="stay":
            action_index = 2
        else:
            action_index = random.randint(0,len(self.__ACTION_SPACE__)-1)

        # Step ball
        self.ball.x += self.BALL_SPEED_X
        self.ball.y += self.BALL_SPEED_Y


        '''
            HANDLE COLLISIONS
        '''
        # IF Ball collision with top/bottom walls
        if self.ball.top <= 3 or self.ball.bottom >= self.HEIGHT-3:
            self.BALL_SPEED_Y *= -1

        # IF Ball out of bounds or collide with paddle
        if (self.ball.right >= self.WIDTH-3) or (self.ball.left <= 5) or self.ball.colliderect(self.agent):
            self.BALL_SPEED_X *= -1


        """OBSERVE NEXT STATE"""
        next_state = self.observe()

        """ Observe Reward"""
        if self.ball.colliderect(self.agent):
            self.cycles += 1
            self.hits += 1
            reward = self.HIT_REWARD
        elif self.ball.left <=5:
            self.cycles += 1
            reward = self.MISS_REWARD
        else:
            diff = abs(next_state[1]-next_state[2])
            reward = (self.HEIGHT -(diff/self.HEIGHT))*(int(self.HIT_REWARD/10))
            reward = self.PASSIVE_REWARD + reward


        self.steps += 1
        return prev_state, action_index, next_state, reward
    
    def init_render(self):
        if self.RENDER_MODE:
            pygame.init()
            self.RENDER_MODE = True
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Atari Ping Pong")
        else:
            pass

    def render(self):
        if self.RENDER_MODE==False:
            self.RENDER_MODE = True
            self.init_render()

        # Clear the screen
        self.screen.fill(BLACK)

        # Draw paddles and ball
        pygame.draw.rect(self.screen, WHITE, self.agent)
        pygame.draw.ellipse(self.screen, WHITE, self.ball)

        # Display scores
        if self.cycles>=1:
            score_text = font.render(f"Performance: {self.hits}/{self.cycles}={self.hits/self.cycles:.2f}", True, WHITE)
            self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 20))

        # Update the display
        pygame.display.flip()
