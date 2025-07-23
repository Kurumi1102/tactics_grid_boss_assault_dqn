import json
import os
from agent_dqn import DQNAgent

PLAYER_ACTION_LOG = "Model/player_actions.csv"
DQN_MODEL_FILE = "Model/dqn_agent_imitation.pt"

# Đọc dữ liệu hành động người chơi
player_data = []
with open(PLAYER_ACTION_LOG, 'r') as f:
    for line in f:
        entry = json.loads(line)
        state = json.loads(entry['state'])
        action = entry['action']
        player_data.append((state, action))

# Map action string về action index
ACTION_MAP_AGENT = {
    0: "normal_attack", 1: "horizontal_shot", 2: "vertical_shot", 3: "heal", 4: "ultimate"
}
ACTION_STR_TO_IDX = {v: k for k, v in ACTION_MAP_AGENT.items()}

def parse_action(action_str):
    for idx, name in ACTION_MAP_AGENT.items():
        if action_str.startswith(name):
            return idx
    return None

# Khởi tạo DQN agent
agent = DQNAgent(model_file=DQN_MODEL_FILE)
agent.epsilon = 0.0  # Không exploration khi imitation

# Huấn luyện DQN từ dữ liệu người chơi
for i in range(len(player_data)-1):
    state, action = player_data[i]
    next_state, _ = player_data[i+1]
    action_idx = parse_action(action)
    if action_idx is not None:
        # Gán reward = 1 cho imitation (hoặc có thể reward = 0)
        agent.learn(state, action_idx, reward=1, next_state_dict=next_state, done=False)

agent.save(DQN_MODEL_FILE)
print(f"Imitation DQN model saved to {DQN_MODEL_FILE}")