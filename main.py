import math, random, copy

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from DGN import DGN, ATT
from config import *
import numpy as np
import pandas as pd
from statistics import mean, stdev
from uas_env import Environment
from buffer import ReplayBuffer
USE_CUDA = torch.cuda.is_available()


def save_model(m, episode):
    torch.save(m.state_dict(), 'models/v5model3episode_'+str(episode)+'.pth')


def load_model(m):
    start = 'models/'
    end = '.pth'
    saves = []
    for file in glob.glob('models/agents5*'):
        print(file)
        _, episode = file[file.find(start)+len(start):file.rfind(end)].split('_')
        saves.append(int(episode))

    if len(saves) > 0:
        latest = max(saves)
        print(saves)
        m.load_state_dict(torch.load('models/agents5episode_'+str(latest)+'.pth'))


env = Environment(n_agents=5,max_agents=5)
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

optimizer = optim.Adam(model.parameters(), lr = 0.0001)
att = ATT(observation_space).cuda()
att_tar = ATT(observation_space).cuda()
att_tar.load_state_dict(att.state_dict())
optimizer_att = optim.Adam(att.parameters(), lr = 0.0001)
criterion = nn.BCELoss()

M_Null = torch.Tensor(np.array([np.eye(max_agents)]*batch_size)).cuda()
M_ZERO = torch.Tensor(np.zeros((batch_size,max_agents,max_agents))).cuda()



n_episodes = 50000
buffer_samples = 200
buffer_samples = 200

results = []


#O = np.ones((batch_size,max_agents,observation_space))
#Next_O = np.ones((batch_size,max_agents,observation_space))
#Matrix = np.ones((batch_size,max_agents,max_agents))
#Next_Matrix = np.ones((batch_size,max_agents,max_agents))

a = 0




test = False
if test:
    path = os.getcwd() + "/scenario/scenarios5/test"
    scenarios = os.listdir(path)
    n_episodes = len(scenarios)

for i_episode in range(n_episodes):
    no_edges = []
    no_edges_atoc = []
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
        if i_episode > 40:
            epsilon -=0.001
            if epsilon < 0.1:
                epsilon = 0.1
    conflicts_timestep = []
    nmacs_timestep = []
    f_conf = -1
    t_loss = 0
    while done == False:

        actions = []
        no_edges.append(adj.sum())

        v_a = np.array(att(torch.Tensor(np.array([observations])).cuda())[0].cpu().data)
        for i in range(n_ant):
            if np.random.rand() < epsilon:
                adj[i] = adj[i] * 0 if np.random.rand() < 0.5 else adj[i] * 1
            else:
                adj[i] = adj[i] * 0 if v_a[i][0] < -0.1 else adj[i] * 1

        adj = adj + np.eye(max_agents)
        no_edges_atoc.append(adj.sum())

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

        if i_episode < 40:
            continue



        for e in range(n_epoch):

            batch = buff.getBatch(batch_size)
            mean_reward = []
            stdev_reward = []



            O, A, R, Next_O, Matrix, Next_Matrix, D = batch
            O = torch.Tensor(O).cuda()
            Matrix = torch.Tensor(Matrix).cuda()
            Next_O = torch.Tensor(Next_O).cuda()
            Next_Matrix = torch.Tensor(Next_Matrix).cuda()

            for i_a in range(n_ant):
                #r = [x[2][i_a] for x in buff.buffer]
                r = R[:,i]
                mean_reward.append(mean(r))
                stdev_reward.append(np.std(np.array(r)))
                #mean_reward.append(mean(r))
                #stdev_reward.append(np.std(np.array(r)))

            label = model(Next_O, Next_Matrix + M_Null).max(dim=2)[0] - model(Next_O, M_Null).max(dim=2)[0]
            label = (label - label.mean()) / (label.std() + 0.000001) + 0.5
            label = torch.clamp(label, 0, 1).unsqueeze(-1).detach()
            loss = criterion(att(Next_O), label)
            optimizer_att.zero_grad()
            loss.backward()
            optimizer_att.step()

            V_A_D = att_tar(Next_O).expand(-1, -1, n_ant)
            Next_Matrix = torch.where(V_A_D > -0.1, Next_Matrix, M_ZERO)
            Next_Matrix = Next_Matrix + M_Null

            q_values = model(O, Matrix)
            target_q_values = model_tar(Next_O, Next_Matrix).max(dim=2)[0]
            target_q_values = np.array(target_q_values.cpu().data)
            expected_q = np.array(q_values.cpu().data)

            for j in range(batch_size):
                for i in range(n_ant):
                    expected_q[j][i][A[j][i]] = (R[j][i] - mean_reward[i])/(stdev_reward[i]+0.001) + (1 - D[j]) * GAMMA * target_q_values[j][i]

            loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for p, p_targ in zip(model.parameters(), model_tar.parameters()):
                    p_targ.data.mul_(tau)
                    p_targ.data.add_((1 - tau) * p.data)
                for p, p_targ in zip(att.parameters(), att_tar.parameters()):
                    p_targ.data.mul_(tau)
                    p_targ.data.add_((1 - tau) * p.data)

            #for j in range(batch_size):
            #    sample = batch[j]
            #    O[j] = sample[0]
            #    Next_O[j] = sample[3]
            #    Matrix[j] = sample[4]
            #    Next_Matrix[j] = sample[5]

            #q_values = model(torch.Tensor(O).cuda(), torch.Tensor(Matrix).cuda())
            ##q_values = model(torch.Tensor(O), torch.Tensor(Matrix))
            #target_q_values = model_tar(torch.Tensor(Next_O).cuda(), torch.Tensor(Next_Matrix).cuda()).max(dim=2)[0]
            ##target_q_values = model_tar(torch.Tensor(Next_O), torch.Tensor(Next_Matrix)).max(dim=2)[0]
            #target_q_values = np.array(target_q_values.cpu().data)
            #expected_q = np.array(q_values.cpu().data)

            #for j in range(batch_size):
            #    sample = batch[j]
            #    for i in range(n_ant):
            #        expected_q[j][i][sample[1][i]] = (sample[2][i]-mean_reward[i])/(stdev_reward[i]+0.01) + (1 - sample[6]) * GAMMA * target_q_values[j][i]

            #loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean()
            #loss = (q_values - torch.Tensor(expected_q)).pow(2).mean()
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()

        #if i_episode % 5 == 0:
        #    model_tar.load_state_dict(model.state_dict())
        if i_episode % 200 == 0:
            df = pd.DataFrame(results)
            df.to_csv('agents5scores_'+str(i_episode)+'.csv')
            save_model(model,i_episode)
if test:
    df = pd.DataFrame(results)
    df.to_csv('test_scores.csv')




