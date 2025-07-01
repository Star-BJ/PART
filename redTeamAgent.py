import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import TextProcess
import random
import ATT_MODEL
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import torch.distributed as dist
import shutil
import json
import bleu
import generator
import gc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device0 = torch.device('cuda:0')
device1 = torch.device('cuda:1')
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

log_directory = "./red_log"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
log_filename = os.path.join(log_directory, "redTeamLog.log")
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            m.bias.data.fill_(0.01)
class Agent(object):
    def __init__(self, config):
        self.v = V_net(config).to(config.DEVICE)
        self.pi = Pi_net(config).to(config.DEVICE)
        self.old_pi = Pi_net(config).to(config.DEVICE)
        self.old_v = V_net(config).to(config.DEVICE)
        self.data = []
        self.step = 0
        self.losses_pi = []
        self.losses_v = []
    def choose_action(self, s):
        with torch.no_grad():
            mu, sigma = self.old_pi(s)
            dis = torch.distributions.normal.Normal(mu, sigma)
            candidate = dis.sample()
        return candidate
    def push_data(self, transitions):
        self.data.append(transitions)
    def sample(self):
        l_s, l_a, l_r, l_s_, l_done = [], [], [], [], []
        for item in self.data:
            s, a, r, s_, done = item
            l_s.append(s.to('cuda:0'))
            l_a.append(a.to('cuda:0'))
            l_r.append(torch.tensor([[r]], dtype=torch.float, device='cuda:0'))
            l_s_.append(s_.to('cuda:0'))
            l_done.append(torch.tensor([[done]], dtype=torch.float, device='cuda:0'))
        s = torch.cat(l_s, dim=0)
        a = torch.cat(l_a, dim=0)
        r = torch.cat(l_r, dim=0)
        s_ = torch.cat(l_s_, dim=0)
        done = torch.cat(l_done, dim=0)
        self.data = []
        return s, a, r, s_, done
    def update(self, config):
        self.step += 1
        s, a, r, s_, done = self.sample()
        s, a, r, s_, done = s.to(config.DEVICE), a.to(config.DEVICE), r.to(config.DEVICE), s_.to(config.DEVICE), done.to(config.DEVICE)
        for _ in range(config.K_epoch):  
            with torch.no_grad():  
                td_target = r + config.GAMMA * self.old_v(s_) * (1 - done)
                mu, sigma = self.old_pi(s)
                old_dis = torch.distributions.normal.Normal(mu, sigma)
                log_prob_old = old_dis.log_prob(a)
                td_error = r + config.GAMMA * self.v(s_) * (1 - done) - self.v(s)
                td_error = td_error.detach().cpu().numpy()
                A = []
                adv = 0.0
                for td in td_error[::-1]:
                    adv = adv * config.GAMMA * config.LAMBDA + td[0]
                    A.append(adv)
                A.reverse()
                A = torch.tensor(A, dtype=torch.float, device=config.DEVICE).reshape(-1, 1)
            mu, sigma = self.pi(s)
            new_dis = torch.distributions.normal.Normal(mu, sigma)
            log_prob_new = new_dis.log_prob(a)
            ratio = torch.exp(log_prob_new - log_prob_old)
            L1 = ratio * A
            L2 = torch.clamp(ratio, 1 - config.CLIP, 1 + config.CLIP) * A
            loss_pi = -torch.min(L1, L2).mean()
            self.pi.optim.zero_grad()
            loss_pi.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.pi.parameters(), max_norm=0.6)
            self.pi.optim.step()
            loss_v = F.mse_loss(td_target.detach(), self.v(s))
            self.v.optim.zero_grad()
            loss_v.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.v.parameters(), max_norm=0.6)
            self.v.optim.step()
        self.losses_pi.append(abs(loss_pi.item()))
        self.losses_v.append(abs(loss_v.item()))    
        self.old_pi.load_state_dict(self.pi.state_dict())
        self.old_v.load_state_dict(self.v.state_dict())
        del s, a, r, s_, done, td_target, td_error, A, mu, sigma, old_dis, log_prob_old, new_dis, log_prob_new, ratio, L1, L2, loss_pi, loss_v
        torch.cuda.empty_cache()
        gc.collect()
    def save(self, ep, SR):
        torch.save(self.pi.state_dict(), f'./Results/NET/pi_{ep}_{SR}.pth')
        torch.save(self.v.state_dict(), f'./Results/NET/v_{ep}_{SR}.pth')
        print(f'...Model saved at step {ep}...')
    def load(self):
        try:
            self.pi.load_state_dict(torch.load('./Results/NET/pi_last.pth'))
            self.v.load_state_dict(torch.load('./Results/NET/v_last.pth'))
            print('...Model loaded...')
        except:
            print('...No saved model found...')
def testLog(ep, rewards_history, success_history, successCase, failCase, losses_pi, losses_v, attNum_history):
        plt.figure()
        plt.plot(range(0,ep+1), rewards_history[:ep+1])
        plt.grid(True)
        plt.title('Training Rewards Over Time')
        plt.xlabel('Number of queries')
        plt.ylabel('Total Reward')
        plt.savefig('./Results/Reward.png')
        plt.close()
        plt.figure()
        plt.plot(attNum_history[:ep+1], success_history[:ep+1])
        plt.grid(True)
        plt.title('Success Over Time')
        plt.xlabel('Number of queries')
        plt.ylabel('Number of positive cases')
        plt.savefig('./Results/SuccessNum.png')
        plt.close()
        if ep > 10:
            plt.figure()
            plt.plot(losses_pi[10:], label='Policy Loss')
            plt.plot(losses_v[10:], label='Value Loss')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Time')
            plt.legend()
            plt.savefig('./Results/Losses.png') 
            plt.close()
        TextProcess.writeCSV('./Results/success/successCase.csv',successCase)
        TextProcess.writeCSV('./Results/success/failCase.csv',failCase)
        data = {"attNum_history": attNum_history, "success_history": success_history}
        with open('./Results/train_history.json', 'w') as file:
            json.dump(data, file)
class StateConstructor:
    def __init__(self, config, embedding_dim=768, window_size=50, hidden_size=64, num_layers=1):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = config.DEVICE
        self.rnn = nn.LSTM(input_size=embedding_dim + 2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True).to(self.device)
        self.fc = nn.Linear(hidden_size, 128).to(self.device)
    def construct_state(self, recent_histories, clo_bleu):
        if len(recent_histories) < self.window_size:
            zero_padding = [(torch.zeros(self.embedding_dim, device=self.device), 0) for _ in range(self.window_size - len(recent_histories))]
            recent_histories = zero_padding + recent_histories
        elif len(recent_histories) > self.window_size:
            recent_histories = recent_histories[-self.window_size:]
        state_components = []
        bleu_tensor = torch.tensor([[clo_bleu]], dtype=torch.float32, device=self.device)
        for i, (vec, success) in enumerate(recent_histories):
            vec = vec.view(1, -1)
            success_tensor = torch.tensor([[float(success)]], dtype=torch.float32, device=self.device)
            state_component = torch.cat([vec, success_tensor, bleu_tensor], dim=1)
            state_components.append(state_component)
        state = torch.cat(state_components, dim=0).view(1, len(recent_histories), -1)
        with torch.no_grad():
            rnn_out, _ = self.rnn(state)
            rnn_out = rnn_out[:, -1, :]
        state = self.fc(rnn_out)
        torch.cuda.empty_cache()
        return state
class Red_PPO():
    def __init__(self, config):
        self.agent = Agent(config)
        self.embedder = TextProcess.SentenceEmbedder()
        self.successCase = list()
        self.failCase = list()
        self.testHistory = list()
        self.testHistoryEmb = list()
        self.successNum = 0
        self.attNum = 0
        self.attNum_history = [0]
        self.max_rewards = -1000000
        self.rewards_history = [0]
        self.success_history = [0]
        self.env = ATT_MODEL.ATT_model()
    def reward_function(self, proba, textNotSafe, clo_bleu):
        r_b = 1.0
        r_d = 1.0 
        base_reward = 1 + abs(proba) if textNotSafe else 1 - abs(proba)
        diversity_bonus = 0
        if textNotSafe:
             diversity_bonus = 1 - clo_bleu
        total_reward = r_b * base_reward + r_d * diversity_bonus
        return total_reward
    def train(self, config):
        candidateSet = TextProcess.readCSVasList('testCasesTemp/candidate_temp.csv')
        stateConstructor = StateConstructor(config=config)
        candidateSet_emb = self.embedder.get_embeddings(candidateSet)
        closeSentenceFinder = TextProcess.closeSentenceFinder(candidateSet_emb)
        rewards = 0
        ca = ''
        for ep in tqdm(range(1, config.EP_MAX+1), desc='Training Progress'):
            done = False
            start_time = time.perf_counter()
            if not self.successCase and not ca:
                clo_bleu = 0
            else:
                cs, clo_bleu = bleu.find_closest_bleu(ca, self.successCase)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            start_time = time.perf_counter()
            s = stateConstructor.construct_state(self.testHistoryEmb, clo_bleu)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time       
            r = 0
            start_time = time.perf_counter()
            a = self.agent.choose_action(s)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time         
            start_time = time.perf_counter()
            ca_index, distance = closeSentenceFinder.find_closest(a)
            ca = candidateSet[ca_index]
            del candidateSet[ca_index]
            closeSentenceFinder.remove_embeddings(ca_index)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            respone, proba, textNotSafe = self.env.ATT(ca)
            ca_embedded = self.embedder.get_embeddings(ca)
            self.testHistoryEmb.append([ca_embedded,textNotSafe])
            start_time = time.perf_counter()
            self.testHistory.append(ca)
            r = self.reward_function(proba, textNotSafe, clo_bleu)
            self.attNum += 1
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            if self.attNum == 90 or (self.attNum > 50 and self.attNum % 400 == 0):
                GT = generator.TextGenerator("mistralai/Mixtral-8x7B-Instruct-v0.1")
                start_time = time.perf_counter()
                generator.generator(GT, 800)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                del GT
                torch.cuda.empty_cache()
                gc.collect()
                candidateSet = TextProcess.readCSVasList('testCasesTemp/candidate_temp.csv')
                failCaseSet = set(self.failCase)
                candidateSet = [item for item in candidateSet if item not in failCaseSet]
                candidateSet_emb = self.embedder.get_embeddings(candidateSet)
                closeSentenceFinder = TextProcess.closeSentenceFinder(candidateSet_emb)
            if textNotSafe:
                self.successCase.append(ca)
                self.successNum += 1
            else:
                self.failCase.append(ca)
            rewards += r
            self.agent.push_data((s, self.embedder.get_embeddings(ca).to(device), r, s, done))
            if config.UPDATA_DELAY:
                if ep % 10 == 0:
                   self.agent.update(config)
            if ep % 2000 == 0:
                self.agent.save(self.attNum, SR=(self.successNum/self.attNum))
            self.rewards_history.append(r)
            self.attNum_history.append(self.attNum)
            self.success_history.append(self.successNum)
            if self.max_rewards < rewards:
                self.max_rewards = rewards
            logging.info(f"当前询问次数:{self.attNum}") 
            logging.info(f"成功次数:{self.successNum}") 
            testLog(ep,self.rewards_history,self.success_history,self.successCase, self.failCase, self.agent.losses_pi,self.agent.losses_v, self.attNum_history) 
            if self.attNum > config.EP_MAX:
                logging.info(f"总询问次数达到上限:{config.EP_MAX}")
                break
        logging.info(f"总询问次数:{self.attNum}") 
        self.agent.save(self.attNum, SR=(self.successNum/self.attNum))
        testLog(ep,self.rewards_history,self.success_history,self.successCase, self.failCase, self.agent.losses_pi,self.agent.losses_v, self.attNum_history)
class Config:
    def __init__(self, ep_max=4000, lr_v=1e-5, lr_pi=1e-5, k_epoch=8, gamma=0.99, Lambda=0.95, update_delay = True, clip=0.3, kl_weight = 0.35, device = device0):
        self.EP_MAX = int(ep_max)
        self.LR_v = lr_v
        self.LR_pi = lr_pi
        self.K_epoch = k_epoch
        self.GAMMA = gamma
        self.LAMBDA = Lambda
        self.CLIP = clip
        self.KL = kl_weight
        self.UPDATA_DELAY = update_delay
        self.DEVICE = device
def copy_training_outputs(source_files, source_dirs, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for file in source_files:
        if os.path.exists(file):
            shutil.copy(file, dest_dir)
    for dir in source_dirs:
        if os.path.exists(dir):
            dest_path = os.path.join(dest_dir, os.path.basename(dir))
            shutil.copytree(dir, dest_path)     
if __name__ == '__main__':
    TextProcess.overwrite_csv('testCasesBase/candidate_gender.csv', 'testCasesTemp/candidate_temp.csv')
    config = Config()
    redTeam = Red_PPO(config)
    redTeam.train(config)