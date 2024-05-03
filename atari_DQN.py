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

from atari_class_functions import Agent,DQNAgent
from atari_DQNs import *

# Initialize pygame
pygame.init()

# Set up the screen
WIDTH, HEIGHT = 840, 600
# screen = pygame.display.set_mode((WIDTH, HEIGHT))
# pygame.display.set_caption("Ping Pong")

# Set up colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Set up paddles
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100
paddle1 = pygame.Rect(0, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
# paddle2 = pygame.Rect(WIDTH - 0 - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)

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
    paddle1_agent.update_lr(1E-7)
    paddle1_agent.update_eps_decay_steps(100)
    paddle1_agent.misses = 0

    paddle1_agent.history['score_per_cycle'] = []
    paddle1_agent.history['epoch_performance'] = []
    paddle1_agent.history['reward'] = []
    paddle1_agent.history['epsilon'] = []
    paddle1_agent.history['alpha'] = []
    paddle1_agent.history['gamma'] = []
    paddle1_agent.history['loss'] = []

    paddle1_agent.states = []
    paddle1_agent.actions = []
    paddle1_agent.rewards = []
    paddle1_agent.next_states = []

elif pretrained_agent.strip():
    print("New Agent being spawned")
    paddle1_agent = DQNAgent(
        paddle1,
        environment={'HEIGHT': HEIGHT, 'WIDTH': WIDTH, 'ball': ball},
        learning_rate=1E-7,
        gamma=0.9,
        epsilon=0.3,
        eps_decay_steps=100,
        eps_range=(0.7,0.8),
        neural_net = mlp_1_0,
        Q_file=pretrained_agent,
        training_epochs=10,
        batch_size=150,
        noLossReward_=1,
        replay_size = 2000
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
last_training_step = 0
cycle = 1
episode = 1
action_choice = random.randint(0,2)

target_dqn = copy.deepcopy(paddle1_agent.DQN)

hits = 0
epsilon = []
alpha = []
gamma = []

best_mean_reward = 0
training_loop = 0

# Main game loop
while True:

    steps += 1

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            paddle1_agent.save()
            pygame.quit()
            sys.exit()

    # Take Action
    action_choice = paddle1_agent.act(cycle, training_loop)

    """
        UPDATE ENIRONMENT AFTER CURRENT STATE ACTION: Q(s,a)
    """
    # Move ball
    ball.x += ball_speed_x
    ball.y += ball_speed_y

    # Ball collision with walls
    if ball.top <= 0 or ball.bottom >= HEIGHT:
        ball_speed_y *= -1
    
    # Ball out of bounds
    if ball.left <= 5:
        score1 -= 1
        ball.x = random.randint(0,int(WIDTH*0.85)) - BALL_SIZE // 2
        ball.y = random.randint(0, HEIGHT) - BALL_SIZE // 2
        ball_speed_x *= -1
        cycle += 1
    if ball.right >= WIDTH:
        ball_speed_x *= -1

    # Ball paddle Hit
    if ball.colliderect(paddle1):
        score1 += 1
        cycle += 1
        hits += 1
        ball_speed_x *= -1


    """
        OBSERVE AND TRAIN
    """
    # Observe next state
    paddle1_agent.observe(ball_velocity=[ball_speed_x, ball_speed_y])

    # Compute Q-values for next state and target-Q-value
    hit_ratio = paddle1_agent.compute_targets(score1, cycle)

    # Update Replay Buffer
    if len(paddle1_agent.states)>=paddle1_agent.replay_size:
        paddle1_agent.states.pop(0)
        paddle1_agent.next_states.pop(0)
        paddle1_agent.actions.pop(0)
        paddle1_agent.rewards.pop(0)
    paddle1_agent.states.append(paddle1_agent.prev_state)
    paddle1_agent.next_states.append(paddle1_agent.state)
    paddle1_agent.actions.append(action_choice)
    paddle1_agent.rewards.append(paddle1_agent.reward)

    # Update Training Hyperparameters
    epsilon.append(paddle1_agent.__epsilon__)
    alpha.append(paddle1_agent.__alpha__)
    gamma.append(paddle1_agent.__gamma__)


    if cycle>episode and len(paddle1_agent.states)>=paddle1_agent.replay_size and cycle-episode>=7:
        print("-------------------------------------------------------------------------------------")
        loss = paddle1_agent.train((paddle1_agent.states, paddle1_agent.actions, paddle1_agent.rewards, paddle1_agent.next_states))

        paddle1_agent.history['epoch_performance'].append(np.array(paddle1_agent.history['score_per_cycle']).mean())
        performance = paddle1_agent.history['epoch_performance'][-1]
        paddle1_agent.history['reward'].append(paddle1_agent.session_rewards)
        paddle1_agent.history['epsilon'].append(sum(epsilon)/len(epsilon))
        paddle1_agent.history['alpha'].append(sum(alpha)/len(alpha))
        paddle1_agent.history['gamma'].append(sum(gamma)/len(gamma))
        paddle1_agent.history['loss'].append(loss.item())
        
        training_loop += 1
        print(f"{training_loop} => performance: {paddle1_agent.history['score_per_cycle'][-1]:.2f} | loss: {loss:.2f} | epsilon: {paddle1_agent.__epsilon__:.2f}")
        print(f"hits: {hits} | no_cycles: {cycle-episode} | cycle: {cycle} | epoch performance: {hits/(cycle-episode):.2f} | rewards: {paddle1_agent.session_rewards}" )
        
        # mean_reward = np.array(paddle1_agent.history['score_per_cycle']).mean()
        mean_reward = hits/(cycle-episode)
        if best_mean_reward<mean_reward:
            paddle1_agent.update_target_network()
            best_mean_reward = mean_reward
            print("TARGET NETWORK CHECKPOINT")

        hits = 0
        paddle1_agent.session_rewards = 0
        epsilon = []
        alpha = []
        gamma = []
        episode=cycle
        last_training_step = steps

    # Move paddles
    # if paddle2.top>ball.top:
    #         if paddle2.top > 0:
    #             paddle2.y -= 10
    # elif paddle2.bottom<ball.bottom:
    #     if paddle2.bottom < HEIGHT:
    #         paddle2.y += 10
    # else:
    #     if abs(ball.right-paddle2.left)<=2 or abs(ball.top-paddle2.top)<=2 or abs(ball.bottom-paddle2.bottom<=2):
    #         dir = random.randint(0,1)
    #         if dir==0:
    #             if paddle2.top > 0:
    #                 paddle2.y -= 10
    #         else:
    #             if paddle2.bottom < HEIGHT:
    #                 paddle2.y += 10
    # keys = pygame.key.get_pressed()
    # if keys[pygame.K_UP] and paddle2.top > 0:
    #     paddle2.y -= 10
    # if keys[pygame.K_DOWN] and paddle2.bottom < HEIGHT:
    #     paddle2.y += 10

    """
        Update Ball Removed From Here
    """

    # Clear the screen
    # screen.fill(BLACK)

    # Draw paddles and ball
    # pygame.draw.rect(screen, WHITE, paddle1)
    # pygame.draw.rect(screen, WHITE, paddle2)
    # pygame.draw.ellipse(screen, WHITE, ball)

    # Display scores
    # score_text = font.render(f"Performance: {np.array(paddle1_agent.history['score_per_cycle']).mean():.2f}", True, WHITE)
    # screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 20))

    if cycle == 1000:
        paddle1_agent.save()
        pygame.quit()
        exit()

    # Update the display
    # pygame.display.flip()

    # Cap the frame rate
    clock.tick(3000)

  
