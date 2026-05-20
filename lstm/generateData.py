import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.machine import Machine

def generate_dataset(n_machines=200, sequence_length=20, save_path='data/lstm_training_data.csv'):
    """
    Simulate many machines from start to failure.
    For each timestep, record the last `sequence_length` sensor readings
    and label it with the true time-to-failure (TTF).
    """
    os.makedirs('data', exist_ok=True)

    all_records = []

    for sim_id in range(n_machines):
        machine = Machine(machine_id=sim_id, seed=sim_id * 7)
        history = []

        step = 0
        while not machine.is_failed and step < 600:
            sensors = machine.step()
            history.append([
                sensors['temperature'],
                sensors['vibration'],
                sensors['wear'],
                sensors['health'],
                sensors['hours_since_maintenance'],
            ])
            step += 1

        failure_step = step  # when it failed

        # Now label each timestep with TTF
        for t in range(len(history)):
            if t < sequence_length:
                continue  # need full sequence
            ttf = failure_step - t  # hours remaining until failure
            sequence = history[t - sequence_length:t]  # last 20 readings

            record = {
                'sim_id': sim_id,
                'timestep': t,
                'ttf': ttf,
            }
            # Flatten sequence into columns
            for seq_idx, reading in enumerate(sequence):
                record[f't{seq_idx}_temp'] = reading[0]
                record[f't{seq_idx}_vib'] = reading[1]
                record[f't{seq_idx}_wear'] = reading[2]
                record[f't{seq_idx}_health'] = reading[3]
                record[f't{seq_idx}_hsm'] = reading[4]

            all_records.append(record)

    df = pd.DataFrame(all_records)
    df.to_csv(save_path, index=False)
    print(f"Dataset generated: {len(df)} samples from {n_machines} machine simulations")
    print(f"TTF range: {df['ttf'].min():.0f} to {df['ttf'].max():.0f} hours")
    print(f"Saved to: {save_path}")
    return df

if __name__ == '__main__':
    df = generate_dataset()
    print(df[['sim_id', 'timestep', 'ttf']].head(10))