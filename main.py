import math, random, copy

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from DGN import DGN
from config import *
import numpy as np
import pandas as pd
from statistics import mean, stdev
from uas_env import Environment
from buffer import ReplayBuffer
USE_CUDA = torch.cuda.is_available()


def save_model(m, episode):
    torch.save(m.state_dict(), 'models/model_'+str(episode)+'.pth')


def load_model(m):
    start = 'models/'
    end = '.pth'
    saves = []
    for file in glob.glob('models/model*'):
        print(file)
        _, episode = file[file.find(start)+len(start):file.rfind(end)].split('_')
        saves.append(int(episode))

    if len(saves) > 0:
        latest = max(saves)
        print(saves)
        m.load_state_dict(torch.load('models/model_'+str(latest)+'.pth'))


env = Environment(n_agents=4,max_agents=4)
max_agents = env.max_agents
n_ant = env.ini_n_planes
observation_space = 4#n_ant * 4
n_actions = 3 #move right, left, do nothing

#buff = ReplayBuffer(capacity)
buff = ReplayBuffer(capacity,observation_space,n_actions,n_ant)
model = DGN(max_agents,observation_space,hidden_dim,n_actions)
load_model(model)
model_tar = DGN(max_agents,observation_space,hidden_dim,n_actions)
load_model(model_tar)
model = model.cuda()
model_tar = model_tar.cuda()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)



n_episodes = 50000
buffer_samples = 200
buffer_samples = 200

results = []


O = np.ones((batch_size,max_agents,observation_space))
Next_O = np.ones((batch_size,max_agents,observation_space))
Matrix = np.ones((batch_size,max_agents,max_agents))
Next_Matrix = np.ones((batch_size,max_agents,max_agents))

a = 0




test = False
if test:
    path = os.getcwd() + "/scenario/scenarios5/test"
    scenarios = os.listdir(path)
    n_episodes = len(scenarios)

for i_episode in range(n_episodes):
    score = 0
    episode_results = {}
    print("Starting episode ", i_episode)
    reward = [0] * max_agents
    adj = np.zeros((max_agents,max_agents))

    observations, done,def_confs,def_nmacs, def_t_loss, def_first_t_loss, generated_hdgs = env.reset()
    episode_results["default_conflicts"] = def_confs
    episode_results["default_nmac"] = def_nmacs
    episode_results["default_t_loss"] = def_t_loss
    episode_results["default_first_t_loss"] = def_first_t_loss
    episode_results["generated_hdgs"] = generated_hdgs

    for idx,agent in enumerate(env.agents_id):
        episode_results['agent'+str(idx)] = []
    agents = env.update_agents()
    scores = np.zeros((n_ant), dtype=np.float32)
    num_steps = 0
    done = False
    i_i = 0
    losmemories = {}
    problem = 0
    #print(problem)
    if not test:
        if i_episode > 100:
            epsilon -=0.0004
            if epsilon < 0.1:
                epsilon = 0.1
    conflicts_timestep = []
    nmacs_timestep = []
    f_conf = -1
    t_loss = 0
    while done == False:

        actions = []

        q = model(torch.Tensor(np.array([observations])).cuda(), torch.Tensor(adj).cuda())[0]
        #q = model(torch.Tensor(np.array([observations])), torch.Tensor(adj))[0]
        for i_a in range(n_ant):
            if np.random.rand() < epsilon:
                a = np.random.randint(n_actions)
            else:
                a = q[i_a].argmax().item()
            print("action taken is "+str(a))
            actions.append(a)
        for idx, agent in enumerate(env.agents_id):
            if agent in env.conf_ac:
                episode_results['agent'+str(idx)].append(actions[idx])
            else:
                episode_results['agent'+str(idx)].append(-100)
            #episode_results[agent].append(actions[idx])
        env.act(actions)
        next_observations, reward, next_adj, done,conf_timestep, nmac_timestep, in_loss = env.step()
        if env.first_t_loss != -1:
            f_conf = env.first_t_loss
        conflicts_timestep.append(conf_timestep)
        nmacs_timestep.append(nmac_timestep)

        buff.add(np.array(observations), actions, reward, np.array(next_observations), adj, next_adj, done)
        observations = next_observations
        adj = next_adj
        score += sum(reward)
        print("SCORE " +str(score))
        num_steps += 1
        t_loss += in_loss


    episode_results["final_conf"] = max(conflicts_timestep)
    episode_results["final_nmac"] = max(nmacs_timestep)
    episode_results["time_steps"] = num_steps
    episode_results["acc_reward"] = score
    episode_results["first_t_loss"] = f_conf
    episode_results["t_loss"] = t_loss
    episode_results["no_edges"] = no_edges
    episode_results["no_edges_atoc"] = no_edges_atoc
    results.append(episode_results)


    if not test:

        if i_episode < 100:
            continue



        for e in range(n_epoch):

            batch = buff.getBatch(batch_size)
            mean_reward = []
            stdev_reward = []

            for j in range(batch_size):
                sample = batch[j]
                O[j] = sample[0]
                Next_O[j] = sample[3]
                Matrix[j] = sample[4]
                Next_Matrix[j] = sample[5]

            q_values = model(torch.Tensor(O).cuda(), torch.Tensor(Matrix).cuda())
            ##q_values = model(torch.Tensor(O), torch.Tensor(Matrix))
            target_q_values = model_tar(torch.Tensor(Next_O).cuda(), torch.Tensor(Next_Matrix).cuda()).max(dim=2)[0]
            ##target_q_values = model_tar(torch.Tensor(Next_O), torch.Tensor(Next_Matrix)).max(dim=2)[0]
            #target_q_values = np.array(target_q_values.cpu().data)
            expected_q = np.array(q_values.cpu().data)

            for j in range(batch_size):
                sample = batch[j]
                for i in range(n_ant):
                    expected_q[j][i][sample[1][i]] = (sample[2][i]-mean_reward[i])/(stdev_reward[i]+0.01) + (1 - sample[6]) * GAMMA * target_q_values[j][i]

            loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean()
            loss = (q_values - torch.Tensor(expected_q)).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i_episode % 5 == 0:
            model_tar.load_state_dict(model.state_dict())
        if i_episode % 200 == 0:
            df = pd.DataFrame(results)
            df.to_csv('scores_'+str(i_episode)+'.csv')
            save_model(model,i_episode)
if test:
    df = pd.DataFrame(results)
    df.to_csv('test_scores.csv')




