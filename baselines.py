import numpy as np
import random
from simulator.factory import Factory

# ---------------------------------------------------------------
# BASELINE 1: Random Policy
# ---------------------------------------------------------------
def run_random_baseline(n_episodes=50, seed=200):
    factory = Factory(n_machines=5, seed=seed)
    all_rewards, all_downtimes, all_costs = [], [], []

    for ep in range(n_episodes):
        state = factory.reset()
        total_reward = 0
        for step in range(500):
            actions = [random.randint(0, 3) for _ in range(5)]
            state, reward, done, _ = factory.step(actions)
            total_reward += reward
            if done:
                break
        summary = factory.get_summary()
        all_rewards.append(total_reward)
        all_downtimes.append(summary['total_downtime_hours'])
        all_costs.append(summary['total_maintenance_cost'])

    return {
        'name':         'Random Policy',
        'avg_reward':   round(np.mean(all_rewards), 1),
        'avg_downtime': round(np.mean(all_downtimes), 2),
        'avg_cost':     round(np.mean(all_costs), 1),
        'std_reward':   round(np.std(all_rewards), 1),
    }


# ---------------------------------------------------------------
# BASELINE 2: Preventive — fixed interval, rotating machines
# ---------------------------------------------------------------
def run_preventive_baseline(n_episodes=50, seed=200):
    factory = Factory(n_machines=5, seed=seed)
    all_rewards, all_downtimes, all_costs = [], [], []

    for ep in range(n_episodes):
        state = factory.reset()
        total_reward = 0
        for step in range(500):
            actions = [0] * 5
            machine_to_maintain = (step // 100) % 5
            if step % 100 == 0 and step > 0:
                actions[machine_to_maintain] = 1
            _, reward, done, _ = factory.step(actions)
            total_reward += reward
            if done:
                break
        summary = factory.get_summary()
        all_rewards.append(total_reward)
        all_downtimes.append(summary['total_downtime_hours'])
        all_costs.append(summary['total_maintenance_cost'])

    return {
        'name':         'Preventive (Fixed Interval)',
        'avg_reward':   round(np.mean(all_rewards), 1),
        'avg_downtime': round(np.mean(all_downtimes), 2),
        'avg_cost':     round(np.mean(all_costs), 1),
        'std_reward':   round(np.std(all_rewards), 1),
    }


# ---------------------------------------------------------------
# BASELINE 3: Predictive — proper ML-only (Random Forest)
#
# This is a genuine ML policy, not a rule-based heuristic.
#
# Phase 1 — Data Collection:
#   Run the factory under a mixed exploratory policy for 100 episodes.
#   At each step, record the full state vector (33 dims) and label
#   each machine's "correct" action using domain knowledge:
#     - Machine failed          -> action 2 (defer, let factory handle)
#     - Health < 30             -> action 1 (maintain now)
#     - Health < 50 and HSM>150 -> action 1 (maintain now)
#     - Health < 70 and HSM>200 -> action 2 (defer 24h)
#     - Otherwise               -> action 0 (produce)
#   These labels represent a reasonable expert policy used ONLY
#   to generate supervised training data.
#
# Phase 2 — Model Training:
#   Train a Random Forest classifier (sklearn) on collected
#   (state, action) pairs. The classifier learns to map factory
#   states to maintenance decisions without any RL or rules —
#   purely from the patterns in the training data.
#
# Phase 3 — Evaluation:
#   Deploy the trained Random Forest as the decision policy.
#   It receives the current state vector and predicts actions
#   for all 5 machines. No rules, no thresholds — pure ML inference.
# ---------------------------------------------------------------
def run_ml_predictive_baseline(n_episodes=50, seed=200):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    print("  [ML Baseline] Phase 1: Collecting training data...")

    # --- Phase 1: Data Collection ---
    factory = Factory(n_machines=5, seed=seed)
    X_train, y_train = [], []

    for ep in range(100):  # 100 collection episodes
        state = factory.reset()
        for step in range(500):
            actions = []
            state_vec = state  # 33-dim state vector

            for i, machine in enumerate(factory.machines):
                h   = machine.health
                hsm = machine.hours_since_maintenance

                # Expert labeling logic for training data generation
                if machine.is_failed:
                    a = 0   # continue — emergency repair handled by factory
                elif h < 30:
                    a = 1   # maintain immediately — critically low health
                elif h < 50 and hsm > 150:
                    a = 1   # maintain — degraded AND overdue
                elif h < 70 and hsm > 200:
                    a = 2   # defer 24h — getting risky
                elif h > 80 and random.random() < 0.05:
                    a = 3   # occasional preemptive replacement
                else:
                    a = 0   # continue production

                actions.append(a)
                # Each machine shares the full state vector
                X_train.append(state_vec.tolist())
                y_train.append(a)

            state, _, done, _ = factory.step(actions)
            if done:
                break

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"  [ML Baseline] Collected {len(X_train)} samples")

    # --- Phase 2: Train Random Forest ---
    print("  [ML Baseline] Phase 2: Training Random Forest classifier...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_scaled, y_train)
    train_acc = clf.score(X_scaled, y_train)
    print(f"  [ML Baseline] Random Forest trained | Train accuracy: {train_acc:.3f}")

    # --- Phase 3: Evaluate ML policy ---
    print("  [ML Baseline] Phase 3: Evaluating ML policy...")
    factory2 = Factory(n_machines=5, seed=seed + 999)
    all_rewards, all_downtimes, all_costs = [], [], []

    for ep in range(n_episodes):
        state = factory2.reset()
        total_reward = 0

        for step in range(500):
            # Pure ML inference — no rules, no thresholds
            state_scaled = scaler.transform(
                np.array(state).reshape(1, -1)
            )
            # Predict one action per machine using same state
            # (each machine's action predicted independently)
            actions = []
            for _ in range(5):
                pred = clf.predict(state_scaled)[0]
                actions.append(int(pred))

            state, reward, done, _ = factory2.step(actions)
            total_reward += reward
            if done:
                break

        summary = factory2.get_summary()
        all_rewards.append(total_reward)
        all_downtimes.append(summary['total_downtime_hours'])
        all_costs.append(summary['total_maintenance_cost'])

    return {
        'name':         'Predictive (ML Only — RF)',
        'avg_reward':   round(np.mean(all_rewards), 1),
        'avg_downtime': round(np.mean(all_downtimes), 2),
        'avg_cost':     round(np.mean(all_costs), 1),
        'std_reward':   round(np.std(all_rewards), 1),
    }


# ---------------------------------------------------------------
# BASELINE 4: RL Agent (DQN)
# ---------------------------------------------------------------
def run_rl_baseline(n_episodes=50, seed=200):
    from rlAgent.dqnAgent import DQNAgent
    factory = Factory(n_machines=5, seed=seed)
    agent   = DQNAgent(n_machines=5, state_size=33, action_size=4)
    agent.load()
    agent.epsilon = 0.0

    all_rewards, all_downtimes, all_costs = [], [], []

    for ep in range(n_episodes):
        state = factory.reset()
        total_reward = 0
        for step in range(500):
            actions = agent.select_actions(state)
            state, reward, done, _ = factory.step(actions)
            total_reward += reward
            if done:
                break
        summary = factory.get_summary()
        all_rewards.append(total_reward)
        all_downtimes.append(summary['total_downtime_hours'])
        all_costs.append(summary['total_maintenance_cost'])

    return {
        'name':         'RL Agent (DQN)',
        'avg_reward':   round(np.mean(all_rewards), 1),
        'avg_downtime': round(np.mean(all_downtimes), 2),
        'avg_cost':     round(np.mean(all_costs), 1),
        'std_reward':   round(np.std(all_rewards), 1),
    }


# ---------------------------------------------------------------
# RUN ALL
# ---------------------------------------------------------------
if __name__ == '__main__':
    print("Running all baselines...\n")

    random_b   = run_random_baseline()
    print(f"✓ Random done: {random_b['avg_reward']}\n")

    preventive = run_preventive_baseline()
    print(f"✓ Preventive done: {preventive['avg_reward']}\n")

    predictive = run_ml_predictive_baseline()
    print(f"✓ ML Predictive done: {predictive['avg_reward']}\n")

    rl         = run_rl_baseline()
    print(f"✓ RL Agent done: {rl['avg_reward']}\n")

    results = [random_b, preventive, predictive, rl]

    print("=" * 70)
    print(f"{'Method':<35} {'Avg Reward':>10} {'Avg Downtime':>13} {'Avg Cost':>10}")
    print("=" * 70)
    for r in results:
        print(f"{r['name']:<35} {r['avg_reward']:>10} "
              f"{r['avg_downtime']:>12}h {r['avg_cost']:>10}")
    print("=" * 70)

    np.save('data/baseline_results.npy', np.array(results, dtype=object))
    print("\nSaved to data/baseline_results.npy")
# import numpy as np
# from simulator.factory import Factory


# # ---------------------------------------------------------------
# # BASELINE 1: Random Policy
# # ---------------------------------------------------------------
# def run_random_baseline(n_episodes=50, seed=200):
#     import random
#     factory = Factory(n_machines=5, seed=seed)
#     all_rewards, all_downtimes, all_costs = [], [], []

#     for ep in range(n_episodes):
#         state = factory.reset()
#         total_reward = 0
#         for step in range(500):
#             actions = [random.randint(0, 3) for _ in range(5)]
#             state, reward, done, _ = factory.step(actions)
#             total_reward += reward
#             if done:
#                 break
#         summary = factory.get_summary()
#         all_rewards.append(total_reward)
#         all_downtimes.append(summary['total_downtime_hours'])
#         all_costs.append(summary['total_maintenance_cost'])

#     return {
#         'name': 'Random Policy',
#         'avg_reward':   round(np.mean(all_rewards), 1),
#         'avg_downtime': round(np.mean(all_downtimes), 2),
#         'avg_cost':     round(np.mean(all_costs), 1),
#         'std_reward':   round(np.std(all_rewards), 1),
#     }


# # ---------------------------------------------------------------
# # BASELINE 2: Preventive — fixed interval, one machine at a time
# # More realistic: only maintain one machine per interval
# # ---------------------------------------------------------------
# def run_preventive_baseline(n_episodes=50, seed=200):
#     factory = Factory(n_machines=5, seed=seed)
#     all_rewards, all_downtimes, all_costs = [], [], []

#     for ep in range(n_episodes):
#         state = factory.reset()
#         total_reward = 0
#         for step in range(500):
#             actions = [0] * 5  # default: continue production
#             # Rotate which machine gets maintained every 100 steps
#             machine_to_maintain = (step // 100) % 5
#             if step % 100 == 0 and step > 0:
#                 actions[machine_to_maintain] = 1
#             _, reward, done, _ = factory.step(actions)
#             total_reward += reward
#             if done:
#                 break
#         summary = factory.get_summary()
#         all_rewards.append(total_reward)
#         all_downtimes.append(summary['total_downtime_hours'])
#         all_costs.append(summary['total_maintenance_cost'])

#     return {
#         'name': 'Preventive (Fixed Interval)',
#         'avg_reward':   round(np.mean(all_rewards), 1),
#         'avg_downtime': round(np.mean(all_downtimes), 2),
#         'avg_cost':     round(np.mean(all_costs), 1),
#         'std_reward':   round(np.std(all_rewards), 1),
#     }


# # ---------------------------------------------------------------
# # BASELINE 3: Predictive — LSTM threshold based
# # ---------------------------------------------------------------
# def run_predictive_baseline(n_episodes=50, ttf_threshold=50, seed=200):
#     factory = Factory(n_machines=5, seed=seed)
#     all_rewards, all_downtimes, all_costs = [], [], []

#     for ep in range(n_episodes):
#         state = factory.reset()
#         total_reward = 0
#         for step in range(500):
#             actions = []
#             for i, machine in enumerate(factory.machines):
#                 if machine.is_failed:
#                     actions.append(0)
#                 elif factory.use_lstm and len(machine.sensor_history) >= 20:
#                     ttf = factory.lstm_predictor.predict_ttf(machine.sensor_history)
#                     if ttf is not None and ttf < ttf_threshold:
#                         actions.append(1)
#                     elif ttf is not None and ttf < ttf_threshold * 2:
#                         actions.append(2)
#                     else:
#                         actions.append(0)
#                 else:
#                     actions.append(0)
#             _, reward, done, _ = factory.step(actions)
#             total_reward += reward
#             if done:
#                 break
#         summary = factory.get_summary()
#         all_rewards.append(total_reward)
#         all_downtimes.append(summary['total_downtime_hours'])
#         all_costs.append(summary['total_maintenance_cost'])

#     return {
#         'name': 'Predictive (ML Only)',
#         'avg_reward':   round(np.mean(all_rewards), 1),
#         'avg_downtime': round(np.mean(all_downtimes), 2),
#         'avg_cost':     round(np.mean(all_costs), 1),
#         'std_reward':   round(np.std(all_rewards), 1),
#     }


# # ---------------------------------------------------------------
# # BASELINE 4: RL Agent
# # ---------------------------------------------------------------
# def run_rl_baseline(n_episodes=50, seed=200):
#     from rlAgent.dqnAgent import DQNAgent
#     factory = Factory(n_machines=5, seed=seed)
#     agent = DQNAgent(n_machines=5, state_size=33, action_size=4)
#     agent.load()
#     agent.epsilon = 0.0

#     all_rewards, all_downtimes, all_costs = [], [], []

#     for ep in range(n_episodes):
#         state = factory.reset()
#         total_reward = 0
#         for step in range(500):
#             actions = agent.select_actions(state)
#             state, reward, done, _ = factory.step(actions)
#             total_reward += reward
#             if done:
#                 break
#         summary = factory.get_summary()
#         all_rewards.append(total_reward)
#         all_downtimes.append(summary['total_downtime_hours'])
#         all_costs.append(summary['total_maintenance_cost'])

#     return {
#         'name': 'RL Agent (DQN)',
#         'avg_reward':   round(np.mean(all_rewards), 1),
#         'avg_downtime': round(np.mean(all_downtimes), 2),
#         'avg_cost':     round(np.mean(all_costs), 1),
#         'std_reward':   round(np.std(all_rewards), 1),
#     }


# # ---------------------------------------------------------------
# # RUN ALL
# # ---------------------------------------------------------------
# if __name__ == '__main__':
#     print("Running all baselines...\n")

#     random_b    = run_random_baseline()
#     print(f"✓ Random done")

#     preventive  = run_preventive_baseline()
#     print(f"✓ Preventive done")

#     predictive  = run_predictive_baseline()
#     print(f"✓ Predictive done")

#     rl          = run_rl_baseline()
#     print(f"✓ RL Agent done\n")

#     results = [random_b, preventive, predictive, rl]

#     print("=" * 70)
#     print(f"{'Method':<30} {'Avg Reward':>12} {'Avg Downtime':>13} {'Avg Cost':>10}")
#     print("=" * 70)
#     for r in results:
#         print(f"{r['name']:<30} {r['avg_reward']:>12} "
#               f"{r['avg_downtime']:>12}h {r['avg_cost']:>10}")
#     print("=" * 70)

#     np.save('data/baseline_results.npy', np.array(results, dtype=object))
#     print("\nSaved to data/baseline_results.npy")