import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

ACTION_MAP_AGENT = {
    0: "normal_attack", 1: "horizontal_shot", 2: "vertical_shot", 3: "heal", 4: "ultimate"
}
NUM_ACTIONS = len(ACTION_MAP_AGENT)
STATE_SIZE = 9  # (boss_hp, boss_rage, cd_hshot, cd_vshot, cd_heal, tank, knight, ad, round)

class DQNNet(nn.Module):
    def __init__(self, state_size=STATE_SIZE, num_actions=NUM_ACTIONS, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_actions)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class DQNAgent:
    def __init__(self, state_size=STATE_SIZE, num_actions=NUM_ACTIONS, lr=1e-3, gamma=0.7, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.05, model_file="Model/dqn_agent.pt"):
        self.state_size = state_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model_file = model_file

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNet(state_size, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = []
        self.batch_size = 64
        self.max_memory = 10000

    def _state_to_vec(self, state_dict):
        boss_hp = state_dict["boss_hp"]
        boss_rage = state_dict["boss_rage"]
        cd_hshot = state_dict["skill_cooldowns"]["horizontal_shot"]
        cd_vshot = state_dict["skill_cooldowns"]["vertical_shot"]
        cd_heal = state_dict["skill_cooldowns"]["heal"]
        tank = state_dict["unit_counts"]["Tank"]
        knight = state_dict["unit_counts"]["Knight"]
        ad = state_dict["unit_counts"]["AD"]
        round_idx = state_dict["current_round"]
        return np.array([boss_hp, boss_rage, cd_hshot, cd_vshot, cd_heal, tank, knight, ad, round_idx], dtype=np.float32)

    def choose_action(self, state_dict, available_skill_keys, grid_units_for_targeting):
        state_vec = self._state_to_vec(state_dict)
        available_action_indices = [idx for idx, sk_key in ACTION_MAP_AGENT.items() if sk_key in available_skill_keys]
        if not available_action_indices:
            return None, [], None
        if random.random() < self.epsilon:
            action_idx = random.choice(available_action_indices)
        else:
            state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).detach().cpu().numpy()[0]
            masked_q = np.full(NUM_ACTIONS, -np.inf)
            for idx in available_action_indices:
                masked_q[idx] = q_values[idx]
            action_idx = int(np.argmax(masked_q))
        chosen_skill_key = ACTION_MAP_AGENT[action_idx]
        skill_params = self._get_heuristic_skill_params(chosen_skill_key, grid_units_for_targeting)
        return chosen_skill_key, skill_params, action_idx

    def _get_heuristic_skill_params(self, skill_key, grid_units):
        # Copy logic từ QTableAgent nếu cần
        player_unit_positions=[]; ads_positions=[]; knights_positions=[]; tanks_positions=[]
        grid_h = len(grid_units); grid_w = len(grid_units[0]) if grid_h > 0 else 0
        for r_loop in range(grid_h):
            for c_loop in range(grid_w):
                unit = grid_units[r_loop][c_loop]
                if unit:
                    player_unit_positions.append((r_loop,c_loop))
                    if unit.name=="AD": ads_positions.append((r_loop,c_loop))
                    elif unit.name=="Knight": knights_positions.append((r_loop,c_loop))
                    elif unit.name=="Tank": tanks_positions.append((r_loop,c_loop))
        params = {}
        if skill_key=="normal_attack":
            target_list_normal = []
            if ads_positions: target_list_normal = [random.choice(ads_positions)]
            elif knights_positions: target_list_normal = [random.choice(knights_positions)]
            elif tanks_positions: target_list_normal = [random.choice(tanks_positions)]
            elif player_unit_positions: target_list_normal = [random.choice(player_unit_positions)]
            return target_list_normal
        elif skill_key=="horizontal_shot":
            best_row,max_targets=-1,-1
            for r_idx in range(grid_h):
                count=sum(1 for c_idx in range(grid_w) if grid_units[r_idx][c_idx] and grid_units[r_idx][c_idx].name in ["AD","Knight"])
                if count>max_targets:max_targets=count;best_row=r_idx
            params["line_idx"] = best_row if best_row != -1 else (random.randint(0,grid_h-1) if grid_h > 0 else 0)
            params["direction"] = random.choice(["ltr", "rtl"])
        elif skill_key=="vertical_shot":
            best_col,max_targets=-1,-1
            for c_idx in range(grid_w):
                count=sum(1 for r_idx in range(grid_h) if grid_units[r_idx][c_idx] and grid_units[r_idx][c_idx].name in ["AD","Knight"])
                if count>max_targets:max_targets=count;best_col=c_idx
            params["line_idx"] = best_col if best_col != -1 else (random.randint(0,grid_w-1) if grid_w > 0 else 0)
            params["direction"] = random.choice(["ttb", "btt"])
        elif skill_key=="ultimate":
            target_list_ulti = []
            targets_ulti_temp = ads_positions+knights_positions+tanks_positions; random.shuffle(targets_ulti_temp)
            if len(targets_ulti_temp)<6:
                empty_cells=[(r_loop,c_loop) for r_loop in range(grid_h) for c_loop in range(grid_w) if grid_units[r_loop][c_loop] is None]
                random.shuffle(empty_cells); targets_ulti_temp.extend(empty_cells[:6-len(targets_ulti_temp)])
            target_list_ulti = targets_ulti_temp[:6]
            return target_list_ulti
        return params

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.max_memory:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, state_dict, action_idx, reward, next_state_dict, done):
        state_vec = self._state_to_vec(state_dict)
        next_state_vec = self._state_to_vec(next_state_dict)
        self.remember(state_vec, action_idx, reward, next_state_vec, done)
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            q_next = self.policy_net(next_states).max(1)[0].unsqueeze(1)
            q_target = rewards + self.gamma * q_next * (1 - dones)
        loss = self.loss_fn(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath=None):
        if filepath is None:
            filepath = self.model_file
        torch.save(self.policy_net.state_dict(), filepath)
        print(f"DQN model saved to {filepath}")

    def load(self, filepath=None):
        if filepath is None:
            filepath = self.model_file
        if os.path.exists(filepath):
            self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
            print(f"DQN model loaded from {filepath}")

def get_game_state_for_dqn(game_logic_instance):
    # Copy logic từ get_game_state_for_q_table nếu cần
    boss=game_logic_instance.boss
    grid=game_logic_instance.grid_units
    state_dict={}
    state_dict["boss_hp"]=boss.current_hp
    state_dict["boss_max_hp"]=boss.max_hp
    state_dict["boss_rage"]=boss.current_rage
    state_dict["skill_cooldowns"]={
        "horizontal_shot":boss.skills["horizontal_shot"]["cd_timer"],
        "vertical_shot":boss.skills["vertical_shot"]["cd_timer"],
        "heal":boss.skills["heal"]["cd_timer"],
    }
    unit_counts={"Tank":0,"Knight":0,"AD":0}
    for r_loop in range(game_logic_instance.grid_size):
        for c_loop in range(game_logic_instance.grid_size):
            unit=grid[r_loop][c_loop]
            if unit and unit.name in unit_counts:
                unit_counts[unit.name]+=1
    state_dict["unit_counts"]=unit_counts
    state_dict["current_round"]=game_logic_instance.current_round
    return state_dict