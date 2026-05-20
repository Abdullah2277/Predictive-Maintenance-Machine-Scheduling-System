import numpy as np
from simulator.factory import Factory
from rlAgent.dqnAgent import DQNAgent

def train_dqn(n_episodes=300, max_steps=500, save_path='models/'):

    factory = Factory(n_machines=5, seed=42)
    agent   = DQNAgent(n_machines=5, state_size=33, action_size=4)

    episode_rewards   = []
    episode_downtimes = []
    best_reward = -np.inf

    print("Starting DQN Training...")
    print(f"Episodes: {n_episodes} | Max steps per episode: {max_steps}\n")

    for episode in range(n_episodes):
        state        = factory.reset()
        total_reward = 0
        total_loss   = 0
        loss_count   = 0

        for step in range(max_steps):
            actions    = agent.select_actions(state)
            next_state, reward, done, info = factory.step(actions)

            agent.store_experience(state, actions, reward, next_state, done)
            loss = agent.learn()

            if loss is not None:
                total_loss += loss
                loss_count += 1

            total_reward += reward
            state = next_state

            if done:
                break

        summary = factory.get_summary()
        episode_rewards.append(total_reward)
        episode_downtimes.append(summary['total_downtime_hours'])
        avg_loss = total_loss / loss_count if loss_count > 0 else 0

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(save_path)

        if (episode + 1) % 10 == 0:
            avg_reward   = np.mean(episode_rewards[-10:])
            avg_downtime = np.mean(episode_downtimes[-10:])
            epsilon      = agent.epsilon
            print(f"Episode {episode+1:3d}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:7.1f} | "
                  f"Avg Downtime: {avg_downtime:.1f}h | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Avg Loss: {avg_loss:.5f}")

    print(f"\nTraining complete. Best reward: {best_reward:.1f}")
    np.save('data/episode_rewards.npy',   np.array(episode_rewards))
    np.save('data/episode_downtimes.npy', np.array(episode_downtimes))
    print("Training history saved.")

    return agent, episode_rewards, episode_downtimes


if __name__ == '__main__':
    train_dqn()