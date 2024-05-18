import os
import copy
import random
import numpy as np
import time

import joblib
from itertools import product
import pandas
# import pygame

import torch
import torch.nn as nn

# torch.manual_seed(42)

from atari_DQNs import mlp_1_0,CustomMSELoss

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def aqn_set_state(environment):
    # set __yp__ (y position of paddle)
    if environment.agent.top>=0 and environment.agent.top<(environment.HEIGHT//5):
        __yp__ = 0
    elif environment.agent.top<=(2*(environment.HEIGHT//5)):
        __yp__ = 1
    elif environment.agent.top<=(3*(environment.HEIGHT//5)):
        __yp__ = 2
    elif environment.agent.top<=(4*(environment.HEIGHT//5)):
        __yp__ = 3
    else:
        __yp__ = 4

    # set __xb__ (x position of ball)
    if environment.ball.right>=0 and environment.ball.right<(environment.WIDTH//5):
        __xb__ = 0
    elif environment.ball.right<=(2*(environment.WIDTH//5)):
        __xb__ = 1
    elif environment.ball.right<=(3*(environment.WIDTH//5)):
        __xb__ = 2
    elif environment.ball.right<=(4*(environment.WIDTH//5)):
        __xb__ = 3
    else:
        __xb__ = 4

    # set __yb__ (y position of the ball)
    if environment.ball.top>=0 and environment.ball.top<(environment.HEIGHT//5):
        __yb__ = 0
    elif environment.ball.top<=(2*(environment.HEIGHT//5)):
        __yb__ = 1
    elif environment.ball.top<=(3*(environment.HEIGHT//5)):
        __yb__ = 2
    elif environment.ball.top<=(4*(environment.HEIGHT//5)):
        __yb__ = 3
    else:
        __yb__ = 4

    # set __bxv__ (x velocity of ball)
    if environment.BALL_SPEED_X>0:
        __bxv__ = 1
    else:
        __bxv__ = 0

    # set __byv__ (y velocity of ball)
    if environment.BALL_SPEED_Y>0:
        __byv__ = 1
    else:
        __byv__ = 0

    __xb__ = float(__xb__)
    __yb__ = float(__yb__)
    __yp__ = float(__yp__)
    __bxv__ = float(__bxv__)
    __byv__ = float(__byv__)

    return (__xb__, __yb__, __yp__, __bxv__, __byv__)

class Agent():
    def __init__(self, agent, environment_params,epsilon=0.1,alpha=0.1,gamma=0.1,Q_file=' ',pos_th=(1,1),noLossReward=0.7,\
                 eps_decay_steps=2000000,eps_range=(1E-6,1)):
        self.__epsilon__ = epsilon
        self.__alpha__ = alpha
        self.__gamma__ = gamma

        self.__xb__ =  float(random.randint(0,4))
        self.__yb__ = float(random.randint(0,4))
        self.__yp__ = float(random.randint(0,4))
        self.__bxv__ = float([0,1][random.randint(0,1)])
        self.__byv__ = float([0,1][random.randint(0,1)])
        self.__states__ = list(product(list(range(0,5)),list(range(0,5)),list(range(0,5)),list(range(0,2)),list(range(0,2))))
        self.__actions__ = ['up','down','stay']
        self.__xth__ = pos_th[0]
        self.__yth__ = pos_th[1]
        self.__noLossReward__ = noLossReward
        self.history = {
            "score_per_cycle":[],
            "epoch_performance":[],
            "reward":[],
            "epsilon":[],
            "alpha":[],
            "gamma":[],
        }
        self.eps_decay_steps = eps_decay_steps
        self.eps_min = eps_range[0]
        self.eps_max = eps_range[1]
        
        self.Q_file=Q_file
        self.Q = np.zeros((len(self.__states__),len(self.__actions__)))
        self.Q = softmax(self.Q)
        self.agent = agent
        self.prev_state = (self.__xb__,self.__yb__,self.__yp__,self.__bxv__,self.__byv__)
        self.state = (self.__xb__,self.__yb__,self.__yp__,self.__bxv__,self.__byv__)
        self.environment = environment_params
        self.score = 0
        self.reward = 0
        self.session_rewards = 0
        self.misses = 0
    
    def update_eps_decay_steps(self,steps):
        self.eps_decay_steps = steps

    def check_states(self):
        return self.__states__
    
    def check_Q_values(self):
        return self.Q

    def check_actions(self):
        return self.__actions__
    
    def set_state(self,ball_velocity):
        # set __yp__ (y position of paddle)
        if self.agent.top>=0 and self.agent.top<(self.environment['HEIGHT']//5):
            self.__yp__ = 0
        elif self.agent.top<=(2*(self.environment['HEIGHT']//5)):
            self.__yp__ = 1
        elif self.agent.top<=(3*(self.environment['HEIGHT']//5)):
            self.__yp__ = 3
        elif self.agent.top<=(4*(self.environment['HEIGHT']//5)):
            self.__yp__ = 3
        else:
            self.__yp__ = 4

        # set __xb__ (x position of ball)
        if self.environment['ball'].right>=0 and self.environment['ball'].right<(self.environment['WIDTH']//5):
            self.__xb__ = 0
        elif self.environment['ball'].right<=(2*(self.environment['WIDTH']//5)):
            self.__xb__ = 1
        elif self.environment['ball'].right<=(3*(self.environment['WIDTH']//5)):
            self.__xb__ = 2
        elif self.environment['ball'].right<=(4*(self.environment['WIDTH']//5)):
            self.__xb__ = 3
        else:
            self.__xb__ = 4

        # set __yb__ (y position of the ball)
        if self.environment['ball'].top>=0 and self.environment['ball'].top<(self.environment['HEIGHT']//5):
            self.__yb__ = 0
        elif self.environment['ball'].top<=(2*(self.environment['HEIGHT']//5)):
            self.__yb__ = 1
        elif self.environment['ball'].top<=(3*(self.environment['HEIGHT']//5)):
            self.__yb__ = 2
        elif self.environment['ball'].top<=(4*(self.environment['HEIGHT']//5)):
            self.__yb__ = 3
        else:
            self.__yb__ = 4 

        # set __bxv__ (x velocity of ball)
        if ball_velocity[0]>0:
            self.__bxv__ = 1
        else:
            self.__bxv__ = 0

        # set __byv__ (y velocity of ball)
        if ball_velocity[1]>0:
            self.__byv__ = 1
        else:
            self.__byv__ = 0

        self.__xb__ = float(self.__xb__)
        self.__yb__ = float(self.__yb__)
        self.__yp__ = float(self.__yp__)
        self.__bxv__ = float(self.__bxv__)
        self.__byv__ = float(self.__byv__)
        
        # set state
        self.prev_state = self.state
        self.state =  (self.__xb__,self.__yb__,self.__yp__,self.__bxv__,self.__byv__)

    def take_action(self,index):
        assert index<=(len(self.__actions__)-1)
        action = self.__actions__[index]
        if action=='up' and self.agent.top>(0.15*self.environment['HEIGHT']):
            self.agent.y-=15
            action_index = 0
        elif action=='down' and self.agent.bottom<(0.85*self.environment['HEIGHT']):
            self.agent.y+=15
            action_index = 1
        else:
            action_index = 2
        return action_index

    def compute_reward(self,current_score,cycle):
        self.reward = current_score - self.score
        if self.reward==0:
            if abs(self.__yb__-self.__yp__)<=2:
                self.reward = self.__noLossReward__
                # current_score += self.reward
        if self.reward==-1:
            self.misses += 1
        hit_ratio = (cycle-self.misses)/cycle
        self.session_rewards += self.reward
        self.history['score_per_cycle'].append(hit_ratio)
        self.score = current_score
        return hit_ratio

    def epsilon_greedy_selection(self):
        if (self.__xb__<=self.__xth__) and (self.__bxv__==0):
            if abs(self.__yb__-self.__yp__)<=self.__yth__:
                eps_min = self.eps_min
                eps_max = self.eps_max
                epsilon= max(eps_min, eps_max - (eps_max-eps_min) * len(self.history["reward"])/self.eps_decay_steps)
                self.__epsilon__ = epsilon
                n = random.random()
                if n <= self.__epsilon__:
                    action_index = random.randint(0,len(self.__actions__)-1)
                else:
                    q = self.__states__.index(self.state)
                    q = self.Q[q,:]
                    action_index = np.argmax(q)
            else:
                action_index = random.randint(0,len(self.__actions__)-1)
        else:
            action_index = 2
        return action_index
    
    def update_Q_values(self,current_score,_cycle_,action_index):
        psi = self.__states__.index(self.prev_state) # previous state index
        nsi = self.__states__.index(self.state) #next/current state index
        hit_ratio = self.compute_reward(current_score=current_score,cycle=_cycle_)
        self.Q[psi,action_index] = self.Q[psi,action_index] + self.__alpha__*(self.reward+(self.__gamma__*max(self.Q[nsi,:]))-self.Q[psi,action_index])
        return hit_ratio

    def save(self):
        joblib.dump(self,self.Q_file)

    def update_agent(self,agent):
        self.agent=agent

class DQNAgent(Agent):

    def __init__(self,agent,environment,learning_rate=0.01,gamma=0.1,epsilon=0.3,eps_decay_steps=2000000,eps_range=(1E-6,1),neural_net=mlp_1_0,Q_file=' ',\
                 training_epochs=7,batch_size=7,noLossReward_=1,replay_size=2000):
        super().__init__(agent,environment,noLossReward=noLossReward_,epsilon=epsilon,alpha=0.3,gamma=gamma,Q_file=Q_file,eps_decay_steps=eps_decay_steps,\
                         eps_range=eps_range)
        self.lr = learning_rate
        self.gamma = gamma
        self.DQN = neural_net()
        self.DQN_target = neural_net()
        self.optimizer = torch.optim.Adam(self.DQN.parameters(),lr=self.lr)
        self.loss = nn.MSELoss()
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.replay_size = replay_size

        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.history['loss'] = []

    def update_lr(self,lr):
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.DQN.parameters(),lr=self.lr)

    def observe(self,ball_velocity):
        self.set_state_(ball_velocity=ball_velocity)

    def set_state_(self,ball_velocity):
        self.__yp__ = self.agent.top
        self.__xb__ = self.environment['ball'].right
        self.__yb__ = self.environment['ball'].top
        self.__bxv__ = ball_velocity[0]
        self.__byv__ = ball_velocity[1]

        self.__xb__ = float(self.__xb__)
        self.__yb__ = float(self.__yb__)
        self.__yp__ = float(self.__yp__)
        self.__bxv__ = float(self.__bxv__)
        self.__byv__ = float(self.__byv__)
        
        # set state
        self.prev_state = self.state
        self.state =  (self.__xb__,self.__yb__,self.__yp__,self.__bxv__,self.__byv__)

    def dqn_select_action(self,cycle,training_loop):
        eps_min = self.eps_min
        eps_max = self.eps_max
        epsilon= max(eps_min, eps_max - (eps_max-eps_min) * training_loop/self.eps_decay_steps)
        self.__epsilon__ = epsilon
        n = random.random()
        if n <= self.__epsilon__:
            action_choice = random.randint(0,len(self.__actions__)-1)
        else:
            self.DQN = self.DQN.eval()
            Q_est = self.DQN(self.state)
            action_choice = torch.argmax(Q_est)
        return action_choice

    def act(self,cycle,training_loop):
        action_choice = self.dqn_select_action(cycle,training_loop)
        action_choice = self.take_action(action_choice)
        return action_choice
    
    def compute_targets(self,current_score,_cycle_):
        hit_ratio = self.compute_reward(current_score=current_score,cycle=_cycle_)
        return hit_ratio
    
    def update_target_network(self):
        self.DQN_target.load_state_dict(self.DQN.state_dict())
    
    def train(self,Q):
        states = torch.tensor(Q[0])
        actions = torch.tensor(Q[1])
        rewards = torch.tensor(Q[2])
        next_states = torch.tensor(Q[3])

        batch_loss = 0
        no_epochs = self.training_epochs
        epoch_size = self.batch_size

        for i in range(no_epochs):

            indices = list(np.random.random_integers(0,states.size()[0]-1,epoch_size))
            self.DQN = self.DQN.eval()
            Q = self.DQN(states[indices, ::])
            Qnsa = self.DQN_target(next_states[indices, ::])
            Qt = (torch.max(Qnsa, dim=1))[0]
            Qt = rewards[indices] + Qt
            y = Qnsa.clone()
            
            for i in range(Q.size()[0]):
                y[i,actions[i]] = Qt[i]


            Q = Q.requires_grad_(True)

            self.DQN = self.DQN.train()

            loss = self.loss(Q,y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_loss += loss
        return batch_loss/no_epochs




        # input = Q[0].requires_grad_(True)
        # target = Q[1].clone().detach()
        # print(input.size(),target.size())
        # loss = self.loss(input,target)

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # self.DQN = self.DQN.eval()

        # return loss