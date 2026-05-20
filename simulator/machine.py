import numpy as np

class Machine:
    def __init__(self, machine_id, seed=None):
        self.machine_id = machine_id
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.health = 100.0
        self.hours_since_maintenance = 0
        self.is_failed = False
        self.age = 0  # total hours this machine has ever run
        self.sensor_history = []

    def get_sensors(self):
        """
        Simulate sensor readings. As health drops, readings worsen.
        """
        degradation_factor = (100 - self.health) / 100  # 0 when healthy, 1 when failed

        temperature = 70 + degradation_factor * 30 + self.rng.normal(0, 2)
        vibration = 0.5 + degradation_factor * 4.5 + self.rng.normal(0, 0.1)
        wear = max(0, self.hours_since_maintenance * 0.02 + degradation_factor * 10 + self.rng.normal(0, 0.5))

        return {
            'temperature': round(temperature, 2),
            'vibration': round(vibration, 3),
            'wear': round(wear, 3),
            'health': round(self.health, 2),
            'hours_since_maintenance': self.hours_since_maintenance,
            'is_failed': int(self.is_failed)
        }

    def step(self):
        """
        Advance machine by one timestep (1 hour).
        Health degrades naturally, faster as machine ages.
        """
        if self.is_failed:
            return self.get_sensors()

        # Degradation rate increases slightly with age
        base_degradation = 0.3 + (self.age / 5000) * 0.2
        noise = self.rng.normal(0, 0.05)
        self.health -= max(0, base_degradation + noise)
        self.health = max(0, self.health)

        self.hours_since_maintenance += 1
        self.age += 1

        if self.health <= 0:
            self.health = 0
            self.is_failed = True

        sensors = self.get_sensors()
        self.sensor_history.append(sensors)
        return sensors

    def perform_maintenance(self):
        """
        Maintenance restores health and resets maintenance clock.
        """
        self.health = min(100, self.health + 60)  # partial restore, not always full
        self.hours_since_maintenance = 0
        self.is_failed = False

    def emergency_repair(self):
        """
        Full repair after failure — expensive in real cost terms.
        """
        self.health = 70.0  # never fully restored after catastrophic failure
        self.hours_since_maintenance = 0
        self.is_failed = False

m1 = Machine(101)
for x in range(500):
    if x%20 == 0:
        print(m1.get_sensors())
    m1.step()