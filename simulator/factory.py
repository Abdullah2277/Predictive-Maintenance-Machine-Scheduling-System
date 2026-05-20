import numpy as np
from simulator.machine import Machine
from lstm.predictor import LSTMPredictor

# ------------------------------------------------------------------
# JOB DEFINITIONS
# Three job types with different production values and deadlines.
# The RL agent implicitly schedules jobs by deciding which machines
# to keep running (action 0) vs take offline for maintenance.
# Healthy machines are assigned higher-priority jobs automatically,
# implementing joint maintenance + production scheduling.
# ------------------------------------------------------------------
JOB_TYPES = {
    0: {'name': 'Standard',  'value': 8,  'deadline_cycles': 100, 'color': 'blue'},
    1: {'name': 'Priority',  'value': 15, 'deadline_cycles': 50,  'color': 'orange'},
    2: {'name': 'Urgent',    'value': 25, 'deadline_cycles': 20,  'color': 'red'},
}

# Reward formula coefficients — explicitly defined for academic clarity
# R = alpha*P - beta*D - gamma*M - delta*L + B
ALPHA = 10   # production value multiplier per unit output
BETA  = 150  # downtime penalty per hour (unplanned >> planned cost)
GAMMA_MAINT = 20   # planned maintenance cost
GAMMA_REPLACE = 35 # part replacement cost
DELTA = 1    # deferral risk penalty
BONUS = 30   # production target completion bonus
LATE_PENALTY = 15  # late delivery penalty


class Factory:
    """
    5-machine factory environment for joint maintenance and
    production scheduling via Reinforcement Learning.

    Reward Formula:
        R = alpha*P - beta*D - gamma*M - delta*L + B
        alpha=10, beta=150, gamma=20/35, delta=1, B=30

    Job Scheduling:
        Machines are automatically assigned jobs based on health:
          - Health >= 70: eligible for Urgent jobs (value=25)
          - Health >= 40: eligible for Priority jobs (value=15)
          - Health <  40: Standard jobs only (value=8)
        The RL agent performs implicit job scheduling by deciding
        which machines stay online (action 0) and which go offline
        for maintenance, directly controlling which job types
        can be fulfilled each timestep.
    """

    def __init__(self, n_machines=5, seed=42):
        self.n_machines = n_machines
        self.rng = np.random.default_rng(seed)
        self.machines = [Machine(machine_id=i, seed=seed+i)
                         for i in range(n_machines)]
        try:
            self.lstm_predictor = LSTMPredictor()
            self.use_lstm = True
        except Exception as e:
            print(f"LSTM not loaded, using linear estimate: {e}")
            self.lstm_predictor = None
            self.use_lstm = False
        self.reset()

    # ------------------------------------------------------------------
    # JOB SCHEDULING LOGIC
    # ------------------------------------------------------------------
    def _assign_job(self, machine):
        """
        Assign job type based on machine health.
        Healthy machines handle high-value urgent jobs.
        Degraded machines are downgraded to lower-value jobs.
        This implements health-aware production scheduling.
        """
        if machine.is_failed:
            return None
        if machine.health >= 70:
            return 2  # Urgent
        elif machine.health >= 40:
            return 1  # Priority
        else:
            return 0  # Standard

    def _get_job_production(self, machine):
        """
        Production output depends on:
          - Machine health (degraded machines produce less)
          - Assigned job type (urgent jobs yield more value)
        """
        job_type = self._assign_job(machine)
        if job_type is None:
            return 0, None
        health_factor = machine.health / 100.0
        base_value    = JOB_TYPES[job_type]['value']
        output        = health_factor * base_value
        return output, job_type

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------
    def reset(self):
        for m in self.machines:
            m.reset()

        self.production_backlog       = 50
        self.spare_parts_inventory    = 10
        self.maintenance_crew_available = 2

        self.current_step          = 0
        self.total_downtime        = 0
        self.total_production      = 0
        self.total_maintenance_cost = 0
        self.episode_log           = []

        # Job tracking
        self.jobs_completed = {0: 0, 1: 0, 2: 0}  # Standard/Priority/Urgent
        self.current_job_assignments = {}           # machine_id -> job_type
        self.urgent_jobs_missed = 0

        return self._get_state()

    # ------------------------------------------------------------------
    # STATE
    # ------------------------------------------------------------------
    def _get_state(self):
        """
        State vector: 33 dimensions
        Per machine (5 x 6 = 30):
          [health, hours_since_maint, job_status, ttf_estimate, temp, vibration]
        System level (3):
          [production_backlog, spare_parts, maintenance_crew]
        """
        state = []
        for m in self.machines:
            sensors = m.get_sensors()
            if self.use_lstm:
                ttf_pred = self.lstm_predictor.predict_ttf(m.sensor_history)
                ttf_estimate = ttf_pred if ttf_pred is not None else max(0, m.health / 0.3)
            else:
                ttf_estimate = max(0, m.health / 0.3)
            job = getattr(m, 'job_status', 1)

            state.extend([
                sensors['health'] / 100.0,
                sensors['hours_since_maintenance'] / 500.0,
                job / 2.0,
                ttf_estimate / 400.0,
                sensors['temperature'] / 120.0,
                sensors['vibration'] / 6.0,
            ])

        state.extend([
            min(self.production_backlog, 200) / 200.0,
            self.spare_parts_inventory / 20.0,
            self.maintenance_crew_available / 2.0,
        ])

        return np.array(state, dtype=np.float32)

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------
    def step(self, actions):
        """
        Actions per machine:
          0 = continue production (run assigned job)
          1 = schedule maintenance now
          2 = schedule maintenance in 24h
          3 = replace part preemptively

        Reward Formula:
          R = alpha*P - beta*D - gamma*M - delta*L + B
          alpha=10, beta=150, gamma=20/35, delta=1, B=30
        """
        assert len(actions) == self.n_machines

        reward = 0
        info   = {'events': [], 'job_assignments': {}}
        crew_used = 0

        for i, (machine, action) in enumerate(zip(self.machines, actions)):

            # ACTION 0: Continue production — run assigned job
            if action == 0:
                if machine.is_failed:
                    machine.emergency_repair()
                    self.total_downtime += 1
                    self.total_maintenance_cost += 50
                    reward -= BETA          # β * D
                    info['events'].append(f'M{i}: emergency repair (-{BETA})')
                    info['job_assignments'][i] = 'FAILED'
                else:
                    machine.job_status = 1
                    sensors = machine.step()

                    # Job scheduling: assign and execute job
                    production, job_type = self._get_job_production(machine)
                    self.total_production += production
                    self.production_backlog = max(
                        0, self.production_backlog - production
                    )
                    reward += production * ALPHA   # α * P

                    if job_type is not None:
                        self.jobs_completed[job_type] += 1
                        self.current_job_assignments[i] = job_type
                        info['job_assignments'][i] = JOB_TYPES[job_type]['name']
                    else:
                        info['job_assignments'][i] = 'Idle'

            # ACTION 1: Maintenance now
            elif action == 1:
                if (crew_used < self.maintenance_crew_available
                        and self.spare_parts_inventory > 0):
                    machine.job_status = 2
                    machine.perform_maintenance()
                    self.spare_parts_inventory  -= 1
                    self.total_maintenance_cost += GAMMA_MAINT
                    crew_used += 1
                    reward -= GAMMA_MAINT     # γ * M
                    info['events'].append(f'M{i}: maintenance performed (-{GAMMA_MAINT})')
                    info['job_assignments'][i] = 'Maintenance'
                else:
                    machine.step()
                    info['events'].append(f'M{i}: maintenance unavailable')
                    info['job_assignments'][i] = 'Waiting'

            # ACTION 2: Defer maintenance 24h
            elif action == 2:
                machine.job_status = 1
                machine.step()
                reward -= DELTA           # δ * L
                info['events'].append(f'M{i}: maintenance deferred (-{DELTA})')
                info['job_assignments'][i] = 'Deferred'

            # ACTION 3: Replace part preemptively
            elif action == 3:
                if self.spare_parts_inventory > 0:
                    machine.job_status        = 2
                    machine.health            = min(100, machine.health + 80)
                    machine.hours_since_maintenance = 0
                    machine.is_failed         = False
                    self.spare_parts_inventory  -= 1
                    self.total_maintenance_cost += GAMMA_REPLACE
                    crew_used += 1
                    reward -= GAMMA_REPLACE   # γ * M (replacement)
                    info['events'].append(f'M{i}: part replaced (-{GAMMA_REPLACE})')
                    info['job_assignments'][i] = 'Part Replacement'
                else:
                    machine.step()
                    info['events'].append(f'M{i}: no spare parts')
                    info['job_assignments'][i] = 'Waiting'

        # System-level: production target bonus
        if self.production_backlog <= 0:
            reward += BONUS
            self.production_backlog = 50
            info['events'].append(f'Target met — new order (+{BONUS})')

        # System-level: late delivery penalty
        if self.production_backlog > 150:
            reward -= LATE_PENALTY
            info['events'].append(f'Late delivery penalty (-{LATE_PENALTY})')

        # Spare parts restock every 50 steps
        if self.current_step % 50 == 0 and self.current_step > 0:
            self.spare_parts_inventory = min(
                20, self.spare_parts_inventory + 3
            )
            info['events'].append('Parts restocked (+3)')

        self.current_step += 1
        done = self.current_step >= 500

        next_state = self._get_state()

        self.episode_log.append({
            'step':        self.current_step,
            'reward':      reward,
            'backlog':     self.production_backlog,
            'spare_parts': self.spare_parts_inventory,
            'downtime':    self.total_downtime,
        })

        return next_state, reward, done, info

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    def get_summary(self):
        return {
            'total_production':      round(self.total_production, 2),
            'total_downtime_hours':  self.total_downtime,
            'total_maintenance_cost': self.total_maintenance_cost,
            'final_backlog':         self.production_backlog,
            'spare_parts_remaining': self.spare_parts_inventory,
            'jobs_completed':        self.jobs_completed,
            'urgent_jobs_missed':    self.urgent_jobs_missed,
        }
