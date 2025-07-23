import sys
import os
import csv
from PyQt5.QtWidgets import QApplication
from agent_dqn import DQNAgent, get_game_state_for_dqn
from main import TacticsGridWindow

NUM_EPISODES_TO_TRAIN = 200000
SAVE_AGENT_EVERY_N_EPISODES = 5000
TRAINING_STATS_FILE = "Model/training_stats_dqn.csv"
DQN_MODEL_FILE = "Model/dqn_agent.pt"

def run_training_loop_dqn(window, agent, num_episodes):
    window.is_fast_mode_training = True
    all_episode_rewards = []
    recent_outcomes = []
    file_exists = os.path.isfile(TRAINING_STATS_FILE)
    with open(TRAINING_STATS_FILE, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(['Episode', 'AvgReward', 'WinRate_Boss', 'Epsilon'])
        for e in range(num_episodes):
            window.current_episode_count = e + 1
            current_game_state_dict = window.game.start_new_game()
            episode_reward = 0
            for game_round_num in range(window.game.max_rounds + 2):
                is_game_over, _ = window.game.check_game_over_conditions()
                if is_game_over:
                    break
                # Player turn (random)
                state_dict_after_player, reward_player_phase, done_player_phase = window.execute_player_turn_for_training()
                episode_reward += reward_player_phase
                if done_player_phase:
                    break
                # Boss turn (DQN agent)
                current_state_dict_for_boss_action = state_dict_after_player
                boss_turn_results = window.execute_boss_turn_for_training()
                status_ui_boss_turn, _msg_ui, _anim, next_state_dict_after_boss, reward_for_boss_this_action, done_after_boss, state_dict_boss_acted_on, action_idx_boss_took = boss_turn_results
                episode_reward += reward_for_boss_this_action
                # DQN update
                if action_idx_boss_took is not None and state_dict_boss_acted_on is not None:
                    agent.learn(state_dict_boss_acted_on, action_idx_boss_took, reward_for_boss_this_action, next_state_dict_after_boss, done_after_boss)
                if done_after_boss:
                    break
                status_nr, msg_nr, next_state_dict_new_round = window.execute_next_round_for_training()
                if status_nr == "game_over":
                    break
            boss_won_episode = False
            final_is_game_over, final_message = window.game.check_game_over_conditions()
            if final_is_game_over:
                if window.game.boss.current_hp <= 0:
                    boss_won_episode = False
                else:
                    boss_won_episode = True
            else:
                boss_won_episode = False
            all_episode_rewards.append(episode_reward)
            recent_outcomes.append(1 if boss_won_episode else 0)
            if len(recent_outcomes) > SAVE_AGENT_EVERY_N_EPISODES:
                recent_outcomes.pop(0)
            if (e + 1) % SAVE_AGENT_EVERY_N_EPISODES == 0:
                avg_reward = sum(all_episode_rewards[-SAVE_AGENT_EVERY_N_EPISODES:]) / len(all_episode_rewards[-SAVE_AGENT_EVERY_N_EPISODES:])
                win_rate_boss = sum(recent_outcomes) / len(recent_outcomes) * 100 if recent_outcomes else 0
                log_str = f"Ep {e+1}/{num_episodes}. Avg Reward (last {SAVE_AGENT_EVERY_N_EPISODES}): {avg_reward:.2f}. Win Rate (Boss): {win_rate_boss:.1f}%. Epsilon: {agent.epsilon:.4f}"
                print(log_str)
                csv_writer.writerow([e + 1, f"{avg_reward:.2f}", f"{win_rate_boss:.1f}", f"{agent.epsilon:.4f}"])
                csvfile.flush()
                agent.save(DQN_MODEL_FILE)
    window.is_fast_mode_training = False
    agent.save(DQN_MODEL_FILE)
    print(f"Training finished. DQN model saved to {DQN_MODEL_FILE}")

def main():
    app = QApplication(sys.argv)
    dqn_agent = DQNAgent(model_file=DQN_MODEL_FILE)
    window = TacticsGridWindow(agent_to_use=dqn_agent)
    window.show()
    print("Starting DQN training loop...")
    run_training_loop_dqn(window, dqn_agent, NUM_EPISODES_TO_TRAIN)
    print(f"Training finished. DQN model saved to {DQN_MODEL_FILE}")
    app.quit()

if __name__ == '__main__':
    main()