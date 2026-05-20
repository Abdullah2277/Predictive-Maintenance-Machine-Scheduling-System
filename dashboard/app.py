import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="🏭",
    layout="wide"
)

@st.cache_data
def load_training_data():
    rewards   = np.load('data/episode_rewards.npy')
    downtimes = np.load('data/episode_downtimes.npy')
    return rewards, downtimes

@st.cache_data
def load_baseline_results():
    results = np.load('data/baseline_results.npy', allow_pickle=True)
    return list(results)

# ---------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/factory.png", width=80)
st.sidebar.title("Control Panel")
st.sidebar.markdown("**Predictive Maintenance**")
st.sidebar.markdown("**& Machine Scheduling**")
st.sidebar.divider()
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🤖 RL Training", "⚙️ Live Simulation", "📈 Comparison", "📋 Report"]
)
st.sidebar.divider()
st.sidebar.markdown("**Algorithm:** Deep Q-Network (DQN)")
st.sidebar.markdown("**Predictive Model:** LSTM (TTF ~2h error)")
st.sidebar.markdown("**Course:** Reinforcement Learning")

# ---------------------------------------------------------------
# PAGE 1: OVERVIEW
# ---------------------------------------------------------------
if page == "📊 Overview":
    st.title("🏭 Predictive Maintenance & Machine Scheduling System")
    st.markdown(
        "*An RL agent learns a joint policy for machine scheduling "
        "(which jobs to run when) and maintenance scheduling "
        "(when to service machines) to minimise downtime and maximise production.*"
    )
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Machines", "5", "Active")
    col2.metric("LSTM TTF Error", "~2.0 hrs", "Validation")
    col3.metric("RL Best Reward", "20,824", "↑33% over training")
    col4.metric("RL vs Preventive", "+16% Reward", "Downtime ↓62%")

    st.divider()

    # System architecture
    st.subheader("System Architecture")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("""
**🔧 Manufacturing Simulator**
- 5 machines, degradation model
- Sensors: temperature, vibration, wear
- Health score 0–100
- Failure & repair dynamics
        """)
    with c2:
        st.success("""
**🧠 LSTM Failure Predictor**
- Input: 20-step sensor sequences
- Output: Time-to-Failure (hours)
- Validation error: ~2.0 hours
- Integrated into RL state vector
        """)
    with c3:
        st.warning("""
**🤖 DQN RL Agent**
- State: 33 dimensions
- Actions: 4 per machine
- Replay buffer + target network
- Trained: 500 episodes
        """)

    st.divider()

    # Reward formula — explicitly shown
    st.subheader("Reward Function (Formal Definition)")
    st.latex(r"R = \alpha \cdot P - \beta \cdot D - \gamma \cdot M - \delta \cdot L + B")
    coef_df = pd.DataFrame({
        'Symbol': ['α (alpha)', 'P', 'β (beta)', 'D', 'γ (gamma)', 'M', 'δ (delta)', 'L', 'B'],
        'Value':  [10, '—', 150, '—', '20 / 35', '—', 1, '—', 30],
        'Meaning': [
            'Production value multiplier',
            'Units produced (scales with machine health)',
            'Downtime penalty (β > γ: unplanned > planned cost)',
            'Hours of unplanned downtime',
            'Maintenance cost (service / part replacement)',
            'Maintenance actions taken',
            'Deferral risk penalty',
            'Late delivery penalty (if backlog > 150)',
            'Bonus for completing production target',
        ]
    })
    st.dataframe(coef_df, use_container_width=True, hide_index=True)
    st.caption(
        "β=150 > γ=20 reflects the industry principle that unplanned downtime "
        "costs 3–5× more than scheduled maintenance (Mobley, 2002)."
    )

    st.divider()

    # Job scheduling explanation
    st.subheader("Job Scheduling Policy")
    st.markdown(
        "The RL agent performs **joint maintenance and production scheduling**. "
        "Machines are automatically assigned jobs based on their current health, "
        "implementing a health-aware job allocation strategy:"
    )
    job_df = pd.DataFrame({
        'Job Type':        ['Urgent',   'Priority', 'Standard'],
        'Health Required': ['≥ 70%',    '≥ 40%',    '< 40%'],
        'Production Value':['25 units', '15 units', '8 units'],
        'Deadline':        ['20 cycles','50 cycles','100 cycles'],
        'RL Implication':  [
            'Agent keeps healthy machines running to capture high-value jobs',
            'Agent defers maintenance on mid-health machines',
            'Agent prioritises maintenance — degraded output not worth downtime risk',
        ]
    })
    st.dataframe(job_df, use_container_width=True, hide_index=True)

    st.divider()

    # Action space
    st.subheader("Action Space")
    act_df = pd.DataFrame({
        'Action ID': [0, 1, 2, 3],
        'Action': [
            'Continue Production',
            'Schedule Maintenance Now',
            'Defer Maintenance 24h',
            'Replace Part Preemptively'
        ],
        'Reward Effect': [
            f'+α·P (production value)',
            f'-γ=20 (planned maintenance)',
            f'-δ=1 (deferral risk)',
            f'-γ=35 (part replacement)'
        ],
        'Description': [
            'Machine runs assigned job; output scales with health',
            'Takes machine offline; requires crew + spare part',
            'Keeps running; small penalty for increased failure risk',
            'Near-full health restore; costs spare part'
        ]
    })
    st.dataframe(act_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
# PAGE 2: RL TRAINING
# ---------------------------------------------------------------
elif page == "🤖 RL Training":
    st.title("🤖 DQN Training Progress")

    try:
        rewards, downtimes = load_training_data()
        episodes = list(range(1, len(rewards) + 1))
        window   = 10
        smoothed    = pd.Series(rewards).rolling(window, min_periods=1).mean()
        smoothed_dt = pd.Series(downtimes).rolling(window, min_periods=1).mean()

        st.subheader("Episode Reward — Learning Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=episodes, y=rewards, mode='lines',
            name='Raw Reward', line=dict(color='lightblue', width=1), opacity=0.5
        ))
        fig.add_trace(go.Scatter(
            x=episodes, y=smoothed, mode='lines',
            name=f'{window}-Ep Moving Avg', line=dict(color='royalblue', width=2.5)
        ))
        fig.update_layout(
            xaxis_title="Episode", yaxis_title="Total Reward",
            height=380, hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Downtime Per Episode")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=episodes, y=downtimes, mode='lines',
            name='Raw Downtime', line=dict(color='lightsalmon', width=1), opacity=0.5
        ))
        fig2.add_trace(go.Scatter(
            x=episodes, y=smoothed_dt, mode='lines',
            name=f'{window}-Ep Moving Avg', line=dict(color='crimson', width=2.5)
        ))
        fig2.update_layout(
            xaxis_title="Episode", yaxis_title="Downtime (hours)",
            height=380, hovermode='x unified'
        )
        st.plotly_chart(fig2, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Episodes",       len(rewards))
        c2.metric("Best Reward",          f"{max(rewards):,.0f}")
        c3.metric("Initial Avg (ep1-10)", f"{np.mean(rewards[:10]):,.0f}")
        c4.metric("Final Avg (last 10)",  f"{np.mean(rewards[-10:]):,.0f}")

        improvement = (np.mean(rewards[-10:]) - np.mean(rewards[:10])) / abs(np.mean(rewards[:10])) * 100
        st.success(f"**Reward improvement over training: +{improvement:.1f}%** — confirms the agent learned a better policy.")

    except FileNotFoundError:
        st.error("Training data not found. Run trainDqn.py first.")

# ---------------------------------------------------------------
# PAGE 3: LIVE SIMULATION
# ---------------------------------------------------------------
elif page == "⚙️ Live Simulation":
    st.title("⚙️ Live Factory Simulation")
    st.markdown(
        "Run a live episode and observe the RL agent making joint "
        "maintenance and job scheduling decisions in real time."
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        policy  = st.selectbox("Policy", [
            "RL Agent (DQN)",
            "Preventive (Fixed Interval)",
            "Predictive (ML Only)",
            "Random"
        ])
        n_steps = st.slider("Steps", 50, 500, 300)
        run_btn = st.button("▶ Run Simulation", type="primary")

    if run_btn:
        import random
        from simulator.factory import Factory
        from rlAgent.dqnAgent import DQNAgent

        factory = Factory(n_machines=5, seed=int(np.random.randint(0, 999)))

        if policy == "RL Agent (DQN)":
            agent = DQNAgent(n_machines=5, state_size=33, action_size=4)
            agent.load()
            agent.epsilon = 0.0

        state = factory.reset()
        step_rewards   = []
        machine_health = {i: [] for i in range(5)}
        job_log        = []
        action_log     = []
        ACTION_NAMES   = {0: 'Produce', 1: 'Maintain', 2: 'Defer', 3: 'Replace'}

        progress = st.progress(0)
        status   = st.empty()

        for step in range(n_steps):
            if policy == "RL Agent (DQN)":
                actions = agent.select_actions(state)
            elif policy == "Preventive (Fixed Interval)":
                actions = [0] * 5
                if step % 100 == 0 and step > 0:
                    actions[(step // 100) % 5] = 1
            elif policy == "Predictive (ML Only)":
                actions = []
                for m in factory.machines:
                    if factory.use_lstm and len(m.sensor_history) >= 20:
                        ttf = factory.lstm_predictor.predict_ttf(m.sensor_history)
                        actions.append(1 if ttf and ttf < 50 else
                                       2 if ttf and ttf < 100 else 0)
                    else:
                        actions.append(0)
            else:
                actions = [random.randint(0, 3) for _ in range(5)]

            state, reward, done, info = factory.step(actions)
            step_rewards.append(reward)

            for i, m in enumerate(factory.machines):
                machine_health[i].append(m.health)

            job_assignments = info.get('job_assignments', {})
            action_log.append({
                'Step': step + 1,
                **{f'M{i}': ACTION_NAMES[a] for i, a in enumerate(actions)},
                **{f'M{i} Job': job_assignments.get(i, '—') for i in range(5)},
                'Reward': round(reward, 1)
            })

            progress.progress((step + 1) / n_steps)
            status.text(f"Step {step+1}/{n_steps} | Step Reward: {reward:.1f}")
            if done:
                break

        st.success("Simulation complete!")

        # Machine health chart
        st.subheader("Machine Health Over Time")
        fig = go.Figure()
        colors = ['royalblue', 'crimson', 'green', 'orange', 'purple']
        for i in range(5):
            fig.add_trace(go.Scatter(
                y=machine_health[i], mode='lines',
                name=f'Machine {i}', line=dict(color=colors[i], width=2)
            ))
        fig.add_hline(y=70, line_dash="dot",  line_color="orange",
                      annotation_text="Urgent job threshold (70%)")
        fig.add_hline(y=40, line_dash="dash", line_color="red",
                      annotation_text="Priority job threshold (40%)")
        fig.update_layout(
            xaxis_title="Step", yaxis_title="Health Score",
            height=400, yaxis=dict(range=[0, 105])
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics
        summary = factory.get_summary()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Reward",     f"{sum(step_rewards):,.0f}")
        c2.metric("Total Production", f"{summary['total_production']:.1f}")
        c3.metric("Downtime",         f"{summary['total_downtime_hours']}h")
        c4.metric("Maintenance Cost", f"${summary['total_maintenance_cost']}")

        # Job completion breakdown
        st.subheader("Job Scheduling Summary")
        jobs = summary.get('jobs_completed', {0: 0, 1: 0, 2: 0})
        jc1, jc2, jc3 = st.columns(3)
        jc1.metric("Standard Jobs",  jobs.get(0, 0), "Value: 8/unit")
        jc2.metric("Priority Jobs",  jobs.get(1, 0), "Value: 15/unit")
        jc3.metric("Urgent Jobs",    jobs.get(2, 0), "Value: 25/unit")

        # Decision log
        st.subheader("Decision Log — Actions & Job Assignments (last 15 steps)")
        log_df = pd.DataFrame(action_log).tail(15)
        st.dataframe(log_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
# PAGE 4: COMPARISON
# ---------------------------------------------------------------
elif page == "📈 Comparison":
    st.title("📈 Policy Comparison")

    try:
        results  = load_baseline_results()
        names    = [r['name'] for r in results]
        rewards  = [r['avg_reward'] for r in results]
        downtime = [r['avg_downtime'] for r in results]
        costs    = [r['avg_cost'] for r in results]
        colors   = ['#636EFA', '#EF553B', '#00CC96', '#FFD700']

        st.subheader("Average Reward by Policy (50 evaluation episodes)")
        fig = go.Figure(go.Bar(
            x=names, y=rewards, marker_color=colors,
            text=[f"{r:,.0f}" for r in rewards], textposition='outside'
        ))
        fig.update_layout(height=400, yaxis_title="Avg Reward")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Avg Downtime (hours)")
            fig2 = go.Figure(go.Bar(
                x=names, y=downtime, marker_color=colors,
                text=[f"{d:.2f}h" for d in downtime], textposition='outside'
            ))
            fig2.update_layout(height=350, yaxis_title="Hours")
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            st.subheader("Avg Maintenance Cost")
            fig3 = go.Figure(go.Bar(
                x=names, y=costs, marker_color=colors,
                text=[f"${c:,.0f}" for c in costs], textposition='outside'
            ))
            fig3.update_layout(height=350, yaxis_title="Cost ($)")
            st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Summary Table")
        df = pd.DataFrame({
            'Policy':       names,
            'Avg Reward':   [f"{r:,.1f}" for r in rewards],
            'Avg Downtime': [f"{d:.2f}h"  for d in downtime],
            'Avg Cost':     [f"${c:,.1f}" for c in costs],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Key Findings")
        st.success("""
**RL Agent achieves highest reward — 16% above the next best policy.**

- **Random policy** is worst by far — confirms this is a non-trivial optimisation problem
- **Preventive policy** causes 3.0h downtime — highest of all policies — because fixed-interval
  scheduling does not respond to actual machine condition
- **Predictive (ML-only)** minimises downtime but leaves production value on the table —
  it is too conservative and takes machines offline unnecessarily
- **RL Agent** learns the optimal trade-off: maximise production output while keeping
  machines healthy — achieving highest reward with balanced downtime and cost
        """)

        st.info("""
**On maintenance cost:** The RL agent's higher cost reflects strategic investment —
it performs more preemptive part replacements (action 3) which restore machines to
near-full health, enabling them to fulfil high-value Urgent jobs (25 units vs 8 units
for Standard jobs). This cost is justified by the 16% reward advantage.
        """)

    except Exception as e:
        st.error(f"Could not load results: {e}")

# ---------------------------------------------------------------
# PAGE 5: REPORT
# ---------------------------------------------------------------
elif page == "📋 Report":
    st.title("📋 Cost-Benefit Analysis Report")

    st.subheader("Executive Summary")
    st.markdown("""
    This system implements a Deep Q-Network (DQN) agent that learns a **joint policy**
    for machine scheduling and maintenance scheduling across a 5-machine manufacturing line.
    The LSTM failure predictor provides time-to-failure estimates with ~2.0 hour accuracy,
    which are embedded directly into the RL state vector, making the agent
    **predictive-aware** in its decision making.
    """)

    st.divider()
    st.subheader("Formal Reward Function")
    st.latex(r"R_t = \alpha \cdot P_t - \beta \cdot D_t - \gamma \cdot M_t - \delta \cdot L_t + B")
    st.markdown("""
| Term | Value | Justification |
|------|-------|---------------|
| α (production multiplier) | 10 | Reflects average revenue per production unit |
| β (downtime penalty) | 150 | Unplanned downtime costs 3–5× planned maintenance (Mobley, 2002) |
| γ (maintenance cost) | 20 / 35 | Service cost vs part replacement cost ratio |
| δ (deferral penalty) | 1 | Small risk penalty for postponing maintenance |
| B (target bonus) | 30 | Incentivises throughput — completing production orders |
    """)

    st.divider()
    st.subheader("Job Scheduling Architecture")
    st.markdown("""
    The RL agent performs **implicit job scheduling** through its action decisions:

    - When the agent selects **Action 0 (Produce)** for a machine, the system
      automatically assigns the highest-value job the machine's health can support
    - A machine at health ≥ 70% is assigned **Urgent jobs** (value = 25 units)
    - A machine at health 40–70% handles **Priority jobs** (value = 15 units)
    - A machine below 40% health is downgraded to **Standard jobs** (value = 8 units)

    This means the agent is incentivised to maintain machine health not just to
    avoid failure, but to **keep machines eligible for high-value jobs** — a direct
    implementation of the joint maintenance + production scheduling objective.
    """)

    st.divider()
    st.subheader("Cost-Benefit Analysis")

    try:
        results  = load_baseline_results()
        rl_r     = next(r for r in results if 'DQN' in r['name'])
        prev_r   = next(r for r in results if 'Preventive' in r['name'])
        pred_r   = next(r for r in results if 'Predictive' in r['name'])

        reward_gain_vs_prev = rl_r['avg_reward'] - prev_r['avg_reward']
        reward_gain_vs_pred = rl_r['avg_reward'] - pred_r['avg_reward']
        downtime_reduction  = prev_r['avg_downtime'] - rl_r['avg_downtime']
        extra_cost          = rl_r['avg_cost'] - prev_r['avg_cost']

        c1, c2, c3 = st.columns(3)
        c1.metric("Reward vs Preventive",
                  f"+{reward_gain_vs_prev:,.0f}",
                  f"+{reward_gain_vs_prev/prev_r['avg_reward']*100:.1f}%")
        c2.metric("Downtime Reduction vs Preventive",
                  f"-{downtime_reduction:.2f}h",
                  "per episode")
        c3.metric("Additional Maintenance Cost",
                  f"+${extra_cost:,.0f}",
                  "vs Preventive")

        st.markdown(f"""
        **ROI Interpretation:**
        The RL agent spends an additional **${extra_cost:,.0f}** on maintenance
        compared to the Preventive baseline, but achieves **{reward_gain_vs_prev:,.0f}
        more reward units** — a return ratio of
        **{reward_gain_vs_prev/max(extra_cost,1):.1f}x** on the additional investment.
        Downtime is reduced by **{downtime_reduction:.2f} hours per episode**,
        which at β=150 per hour represents **${downtime_reduction*150:,.0f}**
        in avoided downtime costs per episode.
        """)

    except Exception as e:
        st.warning(f"Could not compute live figures: {e}")

    st.divider()
    st.subheader("Simulator Design Justification")
    st.markdown("""
    The manufacturing simulator uses a **physics-inspired degradation model**:

    - Health degrades at base rate 0.3 units/cycle, accelerating with machine age
    - Sensor readings (temperature, vibration, wear) deteriorate proportionally
      to the degradation factor — consistent with real industrial sensor behaviour
    - Machine lifetimes (~300–350 cycles) are comparable to NASA CMAPSS FD001
      benchmark (128–362 cycles), validating the simulator's physical plausibility
    - The synthetic approach follows standard RL methodology where direct
      real-world training is impractical due to safety and cost constraints
      (sim-to-real transfer)

    **Reference:** Saxena & Goebel (2008), NASA CMAPSS Turbofan Engine Degradation Dataset.
    """)

    st.divider()
    st.subheader("Conclusion")
    st.success("""
    The DQN-based predictive maintenance system outperforms all baselines on total
    reward (+16% vs next best), demonstrating that RL successfully learns a joint
    maintenance and production scheduling policy. The LSTM integration enables
    predictive-aware decisions, and the health-based job assignment implements
    genuine production scheduling — not just maintenance timing.
    """)
    st.divider()
    st.subheader("Download Report")

    report_text = f"""
    PREDICTIVE MAINTENANCE & MACHINE SCHEDULING SYSTEM
    P16 — Reinforcement Learning Course Project
    ================================================

    FORMAL REWARD FUNCTION
    R = alpha*P - beta*D - gamma*M - delta*L + B

    Coefficients:
    alpha = 10  (production value multiplier)
    beta  = 150 (downtime penalty — unplanned > planned cost)
    gamma = 20/35 (maintenance / part replacement cost)
    delta = 1   (deferral risk penalty)
    B     = 30  (production target completion bonus)

    Justification: beta=150 > gamma=20 reflects industry principle
    that unplanned downtime costs 3-5x more than scheduled maintenance.

    JOB SCHEDULING ARCHITECTURE
    Health >= 70%  -> Urgent jobs   (value = 25 units/cycle)
    Health 40-70%  -> Priority jobs (value = 15 units/cycle)
    Health < 40%   -> Standard jobs (value = 8 units/cycle)

    The RL agent performs joint scheduling: by deciding which machines
    stay online (Action 0) vs go offline for maintenance, it directly
    controls which job types are fulfilled each timestep.

    POLICY COMPARISON RESULTS
    Random Policy          : Avg Reward =  3,603  | Downtime = 0.24h
    Preventive (Fixed)     : Avg Reward = 13,724  | Downtime = 3.00h
    Predictive (ML Only)   : Avg Reward = 13,556  | Downtime = 0.12h
    RL Agent (DQN)         : Avg Reward = 15,884  | Downtime = 1.14h

    RL agent achieves +16% reward over next best policy.
    Preventive baseline has 3x higher downtime than RL agent.

    COST-BENEFIT ANALYSIS
    Additional maintenance cost vs Preventive: +$1,048
    Reward gain vs Preventive               : +2,160 units
    Return ratio                            : 2.1x on additional investment
    Avoided downtime cost (at beta=150/hr)  : $279 per episode

    SIMULATOR DESIGN JUSTIFICATION
    Degradation model: health degrades at 0.3 units/cycle base rate,
    accelerating with machine age (nonlinear, matches real patterns).
    Machine lifetimes: 300-350 cycles, comparable to NASA CMAPSS
    FD001 benchmark (128-362 cycles) — physically plausible.
    Approach: sim-to-real methodology, standard in RL research.
    Reference: Saxena & Goebel (2008), NASA CMAPSS Dataset.

    LSTM FAILURE PREDICTOR
    Architecture : 2-layer LSTM, hidden size 64
    Input        : 20-step sensor sequences (temp, vibration, wear)
    Output       : Time-to-failure estimate (hours)
    Validation error: ~2.0 hours
    Integration  : TTF estimate embedded directly in RL state vector

    CONCLUSION
    The DQN agent outperforms all baselines on total reward (+16%),
    successfully learning joint maintenance and production scheduling.
    LSTM integration enables predictive-aware decisions. Health-based
    job assignment implements genuine production scheduling.
    """

    # Encode as downloadable text file
    import base64
    b64 = base64.b64encode(report_text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="P16_Report.txt">📄 Download Report as Text File</a>'
    st.markdown(href, unsafe_allow_html=True)

    # Also offer copy option
    with st.expander("📋 View full report text"):
        st.text(report_text)
