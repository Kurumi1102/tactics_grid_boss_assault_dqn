import numpy as np
import pickle
import random

ACTION_MAP_AGENT = {
    0: "normal_attack", 1: "horizontal_shot", 2: "vertical_shot", 3: "heal", 4: "ultimate"
}
NUM_ACTIONS = len(ACTION_MAP_AGENT)

class QTableAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.7, exploration_rate=1.0, exploration_decay=0.9999, min_exploration_rate=0.05, qtable_file="qtable.pkl"):
        self.lr = learning_rate
        self.gamma = discount_factor  # Ưu tiên reward tức thời
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration_rate
        self.qtable = dict()
        self.qtable_file = qtable_file

    def _state_to_key(self, state_dict):
        # Chuyển state dict thành tuple để làm key cho Q-table
        # (hp, rage, cd_hshot, cd_vshot, cd_heal, tank, knight, ad, round)
        boss_hp = state_dict["boss_hp"]
        boss_rage = state_dict["boss_rage"]
        cd_hshot = state_dict["skill_cooldowns"]["horizontal_shot"]
        cd_vshot = state_dict["skill_cooldowns"]["vertical_shot"]
        cd_heal = state_dict["skill_cooldowns"]["heal"]
        tank = state_dict["unit_counts"]["Tank"]
        knight = state_dict["unit_counts"]["Knight"]
        ad = state_dict["unit_counts"]["AD"]
        round_idx = state_dict["current_round"]
        return (boss_hp, boss_rage, cd_hshot, cd_vshot, cd_heal, tank, knight, ad, round_idx)

    def choose_action(self, state_dict, available_skill_keys, grid_units_for_targeting):
        state_key = self._state_to_key(state_dict)
        available_action_indices = [idx for idx, sk_key in ACTION_MAP_AGENT.items() if sk_key in available_skill_keys]
        if not available_action_indices:
            return None, [], None
        if random.random() < self.epsilon:
            action_idx = random.choice(available_action_indices)
        else:
            q_values = self.qtable.get(state_key, np.zeros(NUM_ACTIONS))
            masked_q = np.full(NUM_ACTIONS, -np.inf)
            for idx in available_action_indices:
                masked_q[idx] = q_values[idx]
            action_idx = int(np.argmax(masked_q))
        chosen_skill_key = ACTION_MAP_AGENT[action_idx]
        skill_params = self._get_heuristic_skill_params(chosen_skill_key, grid_units_for_targeting)
        return chosen_skill_key, skill_params, action_idx

    def _get_heuristic_skill_params(self, skill_key, grid_units):
        # Copy từ agent.py, giữ nguyên logic chọn mục tiêu
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

    def learn(self, state_dict, action_idx, reward, next_state_dict, done):
        state_key = self._state_to_key(state_dict)
        next_state_key = self._state_to_key(next_state_dict)
        if state_key not in self.qtable:
            self.qtable[state_key] = np.zeros(NUM_ACTIONS)
        if next_state_key not in self.qtable:
            self.qtable[next_state_key] = np.zeros(NUM_ACTIONS)
        q_predict = self.qtable[state_key][action_idx]
        q_target = reward  # reward tức thời, gamma=0
        self.qtable[state_key][action_idx] += self.lr * (q_target - q_predict)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath=None):
        if filepath is None:
            filepath = self.qtable_file
        with open(filepath, "wb") as f:
            pickle.dump(self.qtable, f)
        print(f"Q-table saved to {filepath}")

    def load(self, filepath=None):
        if filepath is None:
            filepath = self.qtable_file
        try:
            with open(filepath, "rb") as f:
                self.qtable = pickle.load(f)
            print(f"Q-table loaded from {filepath}")
        except Exception as e:
            print(f"Could not load Q-table: {e}")

def get_game_state_for_q_table(game_logic_instance):
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