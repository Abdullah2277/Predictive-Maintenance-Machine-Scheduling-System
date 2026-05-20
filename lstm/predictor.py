import numpy as np
import torch
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lstm.trainLstm import TTFPredictor

class LSTMPredictor:
    def __init__(self,
                 model_path='models/lstm_ttf.pth',
                 scaler_path='models/lstm_scaler.pkl',
                 sequence_length=20):

        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load scaler and ttf_max
        with open(scaler_path, 'rb') as f:
            saved = pickle.load(f)
        self.scaler = saved['scaler']
        self.ttf_max = saved['ttf_max']

        # Load model
        self.model = TTFPredictor(input_size=5, hidden_size=64, num_layers=2, dropout=0.2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

        print(f"LSTM predictor loaded (TTF max: {self.ttf_max} hours)")

    def predict_ttf(self, sensor_history):
        """
        sensor_history: list of dicts from machine.sensor_history
        Returns: predicted TTF in hours (float), or None if not enough history
        """
        if len(sensor_history) < self.sequence_length:
            return None  # not enough data yet

        # Take last sequence_length readings
        recent = sensor_history[-self.sequence_length:]

        # Extract features in same order as training
        sequence = np.array([
            [r['temperature'], r['vibration'], r['wear'],
             r['health'], r['hours_since_maintenance']]
            for r in recent
        ])  # shape: (20, 5)

        # Normalize using saved scaler
        flat = sequence.reshape(1, -1)           # (1, 100)
        flat_scaled = self.scaler.transform(flat)
        seq_scaled = flat_scaled.reshape(1, self.sequence_length, 5)  # (1, 20, 5)

        # Predict
        tensor = torch.tensor(seq_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred_normalized = self.model(tensor).item()

        ttf_hours = pred_normalized * self.ttf_max
        return max(0, ttf_hours)


if __name__ == '__main__':
    # Quick test
    from simulator.machine import Machine

    predictor = LSTMPredictor()
    machine = Machine(machine_id=999, seed=999)

    print("\nRunning machine and predicting TTF every 20 steps:\n")
    for step in range(330):
        machine.step()
        if step % 20 == 0 and step >= 20:
            ttf_pred = predictor.predict_ttf(machine.sensor_history)
            true_ttf = 330 - step  # approximate true TTF
            print(f"Step {step:3d} | Health: {machine.health:5.1f} | "
                  f"Predicted TTF: {ttf_pred:5.1f}h | "
                  f"Approx True TTF: {true_ttf}h")