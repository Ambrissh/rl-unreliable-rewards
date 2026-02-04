import torch
import torch.nn.functional as F
import torch.optim as optim
import math
import random
from model import DQN
from replay_buffer import ReplayBuffer

class DQNAgent:
    """DQN Agent with training logic"""
    
    def __init__(self, n_observations, n_actions, config, device):
        self.device = device
        self.config = config
        self.n_actions = n_actions
        
        # Networks
        self.policy_net = DQN(n_observations, n_actions, config.HIDDEN_SIZE).to(device)
        self.target_net = DQN(n_observations, n_actions, config.HIDDEN_SIZE).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=config.LR, amsgrad=True)
        
        # Memory
        self.memory = ReplayBuffer(10000)
        
        # Tracking
        self.steps_done = 0
    
    def select_action(self, state):
        """Epsilon-greedy action selection"""
        sample = random.random()
        eps_threshold = self.config.EPS_END + (self.config.EPS_START - self.config.EPS_END) * \
                       math.exp(-1 * self.steps_done / self.config.EPS_DECAY)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
    
    def optimize_model(self):
        """Train the policy network"""
        if len(self.memory) < self.config.BATCH_SIZE:
            return
        
        transitions = self.memory.sample(self.config.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        # Prepare batches
        state_batch = torch.cat(list(states))
        action_batch = torch.cat(list(actions))
        reward_batch = torch.cat(list(rewards))
        
        non_final_mask = torch.tensor([s is not None for s in next_states], 
                                     dtype=torch.bool, device=self.device)
        non_final_next_states = torch.cat([s for s in next_states if s is not None])
        
        # Compute Q(s, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1})
        next_state_values = torch.zeros(self.config.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.config.GAMMA) + reward_batch
        
        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
    
    def soft_update_target_network(self):
        """Soft update of target network"""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.config.TAU + \
                                        target_net_state_dict[key] * (1 - self.config.TAU)
        self.target_net.load_state_dict(target_net_state_dict)
