import sys
import os
import time
import random
import copy

import pygame
import joblib
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from atari_class_functions import Agent,DQNAgent
from atari_DQNs import *

# Initialize pygame
pygame.init()

# Set up the screen
WIDTH, HEIGHT = 840, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ping Pong")

# Set up colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Set up paddles
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100
paddle1 = pygame.Rect(0, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
paddle2 = pygame.Rect(WIDTH - 0 - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)

# Set up ball
BALL_SIZE = 10
ball = pygame.Rect(WIDTH // 2 - BALL_SIZE // 2, HEIGHT // 2 - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)
ball_speed_x = 7
ball_speed_y = 7

# Initialize agent
training_dir = "DQN_training_sessions/"
pretrained_agent = input('path to pretrained_agent:')
pretrained_agent = training_dir+pretrained_agent
if os.path.exists(pretrained_agent):
    print("Using pretrained Agent initialization")
    time.sleep(2)
    paddle1_agent = joblib.load(pretrained_agent)
    paddle1_agent.update_agent(paddle1)
    paddle1_agent.misses = 0
elif pretrained_agent.strip():
    print("New Agent being spawned")
    paddle1_agent = DQNAgent(
        paddle1,
        environment={'HEIGHT': HEIGHT, 'WIDTH': WIDTH, 'ball': ball},
        learning_rate=0.01,
        gamma=0.1,
        epsilon=0.3,
        eps_decay_steps=2000000,
        eps_range=(0.3,1),
        neural_net = mlp_1_2(),
        Q_file=pretrained_agent
    )
else:
    print("Enter directory for saving agent")
    time.sleep(5)
    exit()

# Set up game variables
clock = pygame.time.Clock()
score1 = 0
font = pygame.font.Font(None, 36)

steps = 0
steps_per_sample = 1
cycle = 1
episode = 1
action_choice = random.randint(0,2)

# Main game loop
while True:

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            paddle1_agent.save()
            pygame.quit()
            sys.exit()
            
    paddle1_agent.observe(ball_velocity=[ball_speed_x, ball_speed_y])
    # if paddle1_agent.prev_state!=paddle1_agent.state:
    Q_ns, Q_ns_t, hit_ratio = paddle1_agent.compute_targets(score1, cycle, action_choice)
    paddle1_agent.batch.append((Q_ns,Q_ns_t))
    steps = 0

    if cycle>episode:
        tensor = torch.tensor(paddle1_agent.batch)
        # if tensor.ndim==2:
        loss = paddle1_agent.train((tensor[:,0,:],tensor[:,1,:]))
        print(f"{cycle} -> performance: {hit_ratio:.2f} | loss: {loss:.2f} | epsilon: {paddle1_agent.__epsilon__:.2f}")
        paddle1_agent.batch = []
        episode=cycle

    action_choice = paddle1_agent.act()
    steps += 1

    if paddle2.top>ball.top:
            if paddle2.top > 0:
                paddle2.y -= 10
    elif paddle2.bottom<ball.bottom:
        if paddle2.bottom < HEIGHT:
            paddle2.y += 10
    else:
        if abs(ball.right-paddle2.left)<=2 or abs(ball.top-paddle2.top)<=2 or abs(ball.bottom-paddle2.bottom<=2):
            dir = random.randint(0,1)
            if dir==0:
                if paddle2.top > 0:
                    paddle2.y -= 10
            else:
                if paddle2.bottom < HEIGHT:
                    paddle2.y += 10

    # Move ball
    ball.x += ball_speed_x
    ball.y += ball_speed_y

    # Ball collision with paddles
    if ball.colliderect(paddle1) or ball.colliderect(paddle2):
        ball_speed_x *= -1

    # Ball collision with walls
    if ball.top <= 0 or ball.bottom >= HEIGHT:
        ball_speed_y *= -1
    
    # Ball out of bounds
    if ball.left <= 5:
        score1 -= 1
        ball.x = WIDTH // 2 - BALL_SIZE // 2
        ball.y = HEIGHT // 2 - BALL_SIZE // 2
        ball_speed_x *= -1
        cycle += 1
    if ball.right >= WIDTH:
        ball_speed_x *= -1
    # Ball paddle Hit
    if ball.colliderect(paddle1):
        score1 += 1
        cycle += 1

    # Clear the screen
    screen.fill(BLACK)

    # Draw paddles and ball
    pygame.draw.rect(screen, WHITE, paddle1)
    pygame.draw.rect(screen, WHITE, paddle2)
    pygame.draw.ellipse(screen, WHITE, ball)

    # Display scores
    score_text = font.render(f"Performance: {hit_ratio:.2f}", True, WHITE)
    screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 20))

    if cycle == 1000:
        paddle1_agent.save()
        pygame.quit()
        exit()

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(1000)

  
