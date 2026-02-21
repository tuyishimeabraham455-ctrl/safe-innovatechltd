"""
PredictAI — Pure NumPy LSTM for RUL Prediction
=============================================
Full LSTM from scratch: forward pass, BPTT, Adam optimizer.
No TensorFlow / PyTorch required — runs on ESP32 Python or any server.

Architecture:
  Input  → LSTM(64) → Dropout(0.3) → LSTM(32) → Dense(16,relu) → Dense(1,linear)
"""

import numpy as np
import json
import os
import math
import pickle

np.random.seed(42)

# ─────────────────────────────────────────────────────────
#  ACTIVATION FUNCTIONS
# ─────────────────────────────────────────────────────────
def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

# ─────────────────────────────────────────────────────────
#  LSTM CELL
# ─────────────────────────────────────────────────────────
class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        k = 1.0 / math.sqrt(hidden_size)
        # Combined weight matrix [Wf, Wi, Wg, Wo] — gates x (input+hidden)
        self.Wx = np.random.uniform(-k, k, (4 * hidden_size, input_size))
        self.Wh = np.random.uniform(-k, k, (4 * hidden_size, hidden_size))
        self.b  = np.zeros((4 * hidden_size,))
        # Adam moments
        self.mWx=np.zeros_like(self.Wx); self.vWx=np.zeros_like(self.Wx)
        self.mWh=np.zeros_like(self.Wh); self.vWh=np.zeros_like(self.Wh)
        self.mb =np.zeros_like(self.b);  self.vb =np.zeros_like(self.b)

    def forward(self, x, h_prev, c_prev):
        """x: (input_size,), returns h, c and cache"""
        combined = self.Wx @ x + self.Wh @ h_prev + self.b
        H = self.hidden_size
        f_raw = combined[0*H:1*H]; i_raw = combined[1*H:2*H]
        g_raw = combined[2*H:3*H]; o_raw = combined[3*H:4*H]
        f = sigmoid(f_raw)
        i = sigmoid(i_raw)
        g = tanh(g_raw)
        o = sigmoid(o_raw)
        c = f * c_prev + i * g
        h = o * tanh(c)
        cache = (x, h_prev, c_prev, f, i, g, o, c, combined)
        return h, c, cache

    def backward(self, dh, dc, cache):
        x, h_prev, c_prev, f, i, g, o, c, combined = cache
        H = self.hidden_size
        tanh_c = tanh(c)
        do = dh * tanh_c
        dc += dh * o * (1 - tanh_c**2)
        df = dc * c_prev
        di = dc * g
        dg = dc * i
        dc_prev = dc * f
        # Gate gradients through activations
        df_raw = df * f * (1 - f)
        di_raw = di * i * (1 - i)
        dg_raw = dg * (1 - g**2)
        do_raw = do * o * (1 - o)
        dcombined = np.concatenate([df_raw, di_raw, dg_raw, do_raw])
        dWx = np.outer(dcombined, x)
        dWh = np.outer(dcombined, h_prev)
        db  = dcombined
        dx      = self.Wx.T @ dcombined
        dh_prev = self.Wh.T @ dcombined
        return dx, dh_prev, dc_prev, dWx, dWh, db


# ─────────────────────────────────────────────────────────
#  DENSE LAYER
# ─────────────────────────────────────────────────────────
class Dense:
    def __init__(self, in_size, out_size, activation='linear'):
        k = math.sqrt(2.0 / in_size)
        self.W = np.random.normal(0, k, (out_size, in_size))
        self.b = np.zeros(out_size)
        self.activation = activation
        self.mW=np.zeros_like(self.W); self.vW=np.zeros_like(self.W)
        self.mb=np.zeros_like(self.b); self.vb=np.zeros_like(self.b)
        self._x = None; self._z = None

    def forward(self, x):
        self._x = x
        self._z = self.W @ x + self.b
        if self.activation == 'relu':
            return relu(self._z)
        return self._z  # linear

    def backward(self, dout):
        if self.activation == 'relu':
            dout = dout * relu_grad(self._z)
        dW = np.outer(dout, self._x)
        db = dout
        dx = self.W.T @ dout
        return dx, dW, db


# ─────────────────────────────────────────────────────────
#  FULL LSTM MODEL
# ─────────────────────────────────────────────────────────
class LSTMModel:
    """
    Architecture:
      LSTM(64) → Dropout(0.3) → LSTM(32) → Dense(16,relu) → Dense(1,linear)
    Input: (window, features=3) — [temp, vib, current]
    Output: scalar RUL [0, 100]
    """
    def __init__(self, input_size=3, hidden1=64, hidden2=32):
        self.lstm1  = LSTMCell(input_size, hidden1)
        self.lstm2  = LSTMCell(hidden1,    hidden2)
        self.dense1 = Dense(hidden2, 16, 'relu')
        self.dense2 = Dense(16, 1, 'linear')
        self.dropout_rate = 0.3
        self.t = 0  # Adam step counter
        self.lr = 1e-3
        self.beta1 = 0.9; self.beta2 = 0.999; self.eps = 1e-8
        self.hidden1 = hidden1; self.hidden2 = hidden2
        self.input_size = input_size

    def forward(self, sequence, training=False):
        """sequence: (T, features)"""
        T = len(sequence)
        h1 = np.zeros(self.hidden1); c1 = np.zeros(self.hidden1)
        h2 = np.zeros(self.hidden2); c2 = np.zeros(self.hidden2)
        cache1_list = []; cache2_list = []
        h1_list = [h1]; c1_list = [c1]
        h2_list = [h2]; c2_list = [c2]
        # LSTM1 pass
        for t in range(T):
            h1, c1, cache1 = self.lstm1.forward(sequence[t], h1, c1)
            cache1_list.append(cache1)
            h1_list.append(h1); c1_list.append(c1)
        # Dropout on h1 output (only during training)
        if training:
            self._drop_mask = (np.random.rand(*h1.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            h1_drop = h1 * self._drop_mask
        else:
            h1_drop = h1
            self._drop_mask = None
        # LSTM2 pass on single step (using last hidden from lstm1)
        h2, c2, cache2 = self.lstm2.forward(h1_drop, h2, c2)
        # Dense layers
        d1_out = self.dense1.forward(h2)
        out    = self.dense2.forward(d1_out)
        self._cache = (cache1_list, cache2, h1, h1_drop)
        return float(out[0])

    def backward(self, loss_grad):
        cache1_list, cache2, h1, h1_drop = self._cache
        dout = np.array([loss_grad])
        # Dense backward
        dh2, dW2, db2 = self.dense2.backward(dout)
        dh2, dW1d, db1d = self.dense1.backward(dh2)
        # LSTM2 backward
        dx1_drop, dh2_prev, dc2, dWx2, dWh2, db2_lstm = self.lstm2.backward(dh2, np.zeros(self.hidden2), cache2)
        # Apply dropout mask
        if self._drop_mask is not None:
            dh1 = dx1_drop * self._drop_mask
        else:
            dh1 = dx1_drop
        # LSTM1 BPTT (backprop through all time steps)
        dWx1_total = np.zeros_like(self.lstm1.Wx)
        dWh1_total = np.zeros_like(self.lstm1.Wh)
        db1_total  = np.zeros_like(self.lstm1.b)
        dc1 = np.zeros(self.hidden1)
        for t in reversed(range(len(cache1_list))):
            dx, dh1, dc1, dWx1, dWh1, db1 = self.lstm1.backward(dh1, dc1, cache1_list[t])
            dWx1_total += dWx1; dWh1_total += dWh1; db1_total += db1
        # Adam updates
        self.t += 1
        self._adam(self.lstm1, 'Wx', dWx1_total)
        self._adam(self.lstm1, 'Wh', dWh1_total)
        self._adam_vec(self.lstm1, 'b', db1_total)
        self._adam(self.lstm2, 'Wx', dWx2)
        self._adam(self.lstm2, 'Wh', dWh2)
        self._adam_vec(self.lstm2, 'b', db2_lstm)
        self._adam(self.dense1, 'W', dW1d)
        self._adam_vec(self.dense1, 'b', db1d)
        self._adam(self.dense2, 'W', dW2)
        self._adam_vec(self.dense2, 'b', db2)

    def _adam(self, layer, attr, grad):
        m = getattr(layer, 'm'+attr); v = getattr(layer, 'v'+attr)
        w = getattr(layer, attr)
        m = self.beta1*m + (1-self.beta1)*grad
        v = self.beta2*v + (1-self.beta2)*(grad**2)
        m_hat = m / (1 - self.beta1**self.t)
        v_hat = v / (1 - self.beta2**self.t)
        setattr(layer, attr, w - self.lr * m_hat / (np.sqrt(v_hat)+self.eps))
        setattr(layer, 'm'+attr, m); setattr(layer, 'v'+attr, v)

    def _adam_vec(self, layer, attr, grad):
        self._adam(layer, attr, grad)

    def predict(self, sequence):
        """sequence: (T, 3) numpy array. Returns RUL [0,100]"""
        rul = self.forward(sequence, training=False)
        return max(0.0, min(100.0, rul))

    def save(self, path):
        weights = {
            'lstm1_Wx': self.lstm1.Wx.tolist(),
            'lstm1_Wh': self.lstm1.Wh.tolist(),
            'lstm1_b':  self.lstm1.b.tolist(),
            'lstm2_Wx': self.lstm2.Wx.tolist(),
            'lstm2_Wh': self.lstm2.Wh.tolist(),
            'lstm2_b':  self.lstm2.b.tolist(),
            'd1_W': self.dense1.W.tolist(),
            'd1_b': self.dense1.b.tolist(),
            'd2_W': self.dense2.W.tolist(),
            'd2_b': self.dense2.b.tolist(),
            'meta': {
                'input_size': self.input_size,
                'hidden1': self.hidden1,
                'hidden2': self.hidden2,
                'dropout': self.dropout_rate,
            }
        }
        with open(path, 'w') as f:
            json.dump(weights, f)
        print(f"[✓] Model saved → {path}")

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            w = json.load(f)
        meta = w['meta']
        model = cls(meta['input_size'], meta['hidden1'], meta['hidden2'])
        model.lstm1.Wx = np.array(w['lstm1_Wx'])
        model.lstm1.Wh = np.array(w['lstm1_Wh'])
        model.lstm1.b  = np.array(w['lstm1_b'])
        model.lstm2.Wx = np.array(w['lstm2_Wx'])
        model.lstm2.Wh = np.array(w['lstm2_Wh'])
        model.lstm2.b  = np.array(w['lstm2_b'])
        model.dense1.W = np.array(w['d1_W'])
        model.dense1.b = np.array(w['d1_b'])
        model.dense2.W = np.array(w['d2_W'])
        model.dense2.b = np.array(w['d2_b'])
        print(f"[✓] Model loaded ← {path}")
        return model


# ─────────────────────────────────────────────────────────
#  DATA GENERATOR (simulates NASA C-MAPSS style data)
# ─────────────────────────────────────────────────────────
def generate_training_data(n_units=80, max_cycles=200, window=30):
    """
    Generates synthetic multi-sensor degradation data.
    Mimics NASA C-MAPSS FD001 degradation patterns.
    Returns X (N, window, 3), y (N,) RUL values [0,100]
    """
    X_all, y_all = [], []
    rng = np.random.RandomState(42)

    # Machine type profiles
    profiles = [
        {'T0':45, 'V0':0.4, 'I0':3.0, 'Tk':0.25, 'Vk':0.018, 'Ik':0.020, 'life':180},
        {'T0':50, 'V0':0.5, 'I0':3.5, 'Tk':0.30, 'Vk':0.022, 'Ik':0.025, 'life':160},
        {'T0':42, 'V0':0.35,'I0':2.8, 'Tk':0.20, 'Vk':0.015, 'Ik':0.018, 'life':200},
        {'T0':55, 'V0':0.6, 'I0':4.0, 'Tk':0.35, 'Vk':0.025, 'Ik':0.030, 'life':140},
    ]

    for unit in range(n_units):
        p = profiles[unit % len(profiles)]
        life = p['life'] + rng.randint(-20, 20)
        life = max(60, life)

        # Generate sensor readings over lifetime
        temps  = []
        vibs   = []
        currs  = []

        for c in range(life):
            pct = c / life  # 0→1 progress (degradation)
            # Non-linear degradation: slow start, accelerating end
            deg = pct**1.8

            t_noise = rng.normal(0, 1.2)
            v_noise = rng.normal(0, 0.04)
            i_noise = rng.normal(0, 0.12)

            temp  = p['T0'] + deg * p['Tk'] * life + t_noise + rng.normal(0, 0.5)
            vib   = p['V0'] + deg * p['Vk'] * life + v_noise
            curr  = p['I0'] + deg * p['Ik'] * life + i_noise

            # Occasional spikes (sensor anomalies)
            if rng.rand() < 0.03:
                temp  += rng.uniform(5, 15)
                vib   += rng.uniform(0.3, 1.0)
                curr  += rng.uniform(0.5, 2.0)

            temps.append(max(20, temp))
            vibs.append(max(0.1, vib))
            currs.append(max(0.5, curr))

        # Normalize per-unit (min-max)
        temps_n  = [(v - min(temps))  / (max(temps)  - min(temps)  + 1e-8) for v in temps]
        vibs_n   = [(v - min(vibs))   / (max(vibs)   - min(vibs)   + 1e-8) for v in vibs]
        currs_n  = [(v - min(currs))  / (max(currs)  - min(currs)  + 1e-8) for v in currs]

        # Create sliding windows
        for start in range(life - window):
            seq = np.array([[temps_n[start+i], vibs_n[start+i], currs_n[start+i]]
                             for i in range(window)])
            rul = max(0.0, min(100.0, (life - (start + window)) / life * 100))
            X_all.append(seq)
            y_all.append(rul)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)

    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


# ─────────────────────────────────────────────────────────
#  TRAINING ENGINE
# ─────────────────────────────────────────────────────────
def train_model(epochs=60, batch_size=32, lr=1e-3, save_path='models/rul_model.json'):
    print("="*55)
    print("  PredictAI — LSTM Training Engine")
    print("="*55)

    # Generate data
    print("\n[1/4] Generating training data (NASA C-MAPSS style)...")
    X, y = generate_training_data(n_units=100, max_cycles=200, window=30)
    print(f"      Dataset: {X.shape[0]} samples | window={X.shape[1]} | features={X.shape[2]}")

    # Split 70/15/15
    n = len(X)
    n_train = int(0.70 * n); n_val = int(0.15 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:], y[n_train+n_val:]
    print(f"      Train:{len(X_train)} | Val:{len(X_val)} | Test:{len(X_test)}")

    # Save scaler stats (for inference normalization)
    scaler_stats = {
        'temp_min': 20.0,  'temp_max': 110.0,
        'vib_min':  0.1,   'vib_max':  8.0,
        'curr_min': 0.5,   'curr_max': 12.0,
    }
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    with open(save_path.replace('.json', '_scaler.json'), 'w') as f:
        json.dump(scaler_stats, f)

    # Build model
    print("\n[2/4] Building LSTM model...")
    print("      Architecture: Input(3) → LSTM(64) → Drop(0.3) → LSTM(32) → Dense(16) → Dense(1)")
    model = LSTMModel(input_size=3, hidden1=64, hidden2=32)
    model.lr = lr

    # Training
    print(f"\n[3/4] Training for {epochs} epochs...")
    best_val_loss = float('inf')
    best_weights_path = save_path.replace('.json', '_best.json')
    patience = 12; no_improve = 0
    history = {'train_loss':[], 'val_loss':[], 'train_mae':[], 'val_mae':[]}

    for epoch in range(epochs):
        # Mini-batch SGD
        idx = np.random.permutation(len(X_train))
        train_losses = []; train_maes = []

        for start in range(0, min(len(X_train), 800), batch_size):  # cap per epoch for speed
            batch_idx = idx[start:start+batch_size]
            batch_loss = 0; batch_mae = 0
            for bi in batch_idx:
                pred = model.forward(X_train[bi], training=True)
                target = float(y_train[bi])
                error  = pred - target
                loss   = error ** 2
                batch_loss += loss; batch_mae += abs(error)
                model.backward(2 * error)
            train_losses.append(batch_loss / len(batch_idx))
            train_maes.append(batch_mae / len(batch_idx))

        # Validation
        val_preds  = [model.predict(X_val[i]) for i in range(min(200, len(X_val)))]
        val_targets= [float(y_val[i]) for i in range(min(200, len(X_val)))]
        val_loss   = float(np.mean([(p-t)**2 for p,t in zip(val_preds, val_targets)]))
        val_mae    = float(np.mean([abs(p-t) for p,t in zip(val_preds, val_targets)]))

        avg_train = float(np.mean(train_losses))
        avg_mae   = float(np.mean(train_maes))

        history['train_loss'].append(avg_train)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(avg_mae)
        history['val_mae'].append(val_mae)

        # LR decay
        if epoch > 0 and epoch % 20 == 0:
            model.lr *= 0.5
            print(f"      LR decay → {model.lr:.2e}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            model.save(best_weights_path)
        else:
            no_improve += 1

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"  Ep {epoch+1:3d}/{epochs} | Train MSE:{avg_train:7.2f} MAE:{avg_mae:5.2f} | Val MSE:{val_loss:7.2f} MAE:{val_mae:5.2f} {'★ BEST' if no_improve==0 else ''}")

        if no_improve >= patience:
            print(f"\n  [Early stop] No improvement for {patience} epochs.")
            break

    # Load best weights
    model = LSTMModel.load(best_weights_path)

    # Final evaluation
    print("\n[4/4] Final Evaluation on Test Set...")
    test_preds   = [model.predict(X_test[i]) for i in range(len(X_test))]
    test_targets = [float(y_test[i]) for i in range(len(X_test))]
    mae   = float(np.mean([abs(p-t) for p,t in zip(test_preds, test_targets)]))
    mse   = float(np.mean([(p-t)**2 for p,t in zip(test_preds, test_targets)]))
    rmse  = math.sqrt(mse)
    ss_res = sum((p-t)**2 for p,t in zip(test_preds, test_targets))
    ss_tot = sum((t - np.mean(test_targets))**2 for t in test_targets)
    r2    = 1 - ss_res / (ss_tot + 1e-8)
    acc   = max(0, 100 * (1 - mae / (np.mean(test_targets) + 1e-8)))

    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │  MAE  : {mae:6.2f} cycles         │")
    print(f"  │  RMSE : {rmse:6.2f} cycles         │")
    print(f"  │  R²   : {r2:6.4f}                │")
    print(f"  │  Acc  : {acc:5.1f}%                │")
    print(f"  └─────────────────────────────────┘")

    # Save final model + metrics
    model.save(save_path)
    metrics = {'mae': mae, 'rmse': rmse, 'r2': r2, 'accuracy': acc,
               'history': history, 'test_samples': len(X_test)}
    with open(save_path.replace('.json', '_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  [✓] Model saved  → {save_path}")
    print(f"  [✓] Metrics saved → {save_path.replace('.json','_metrics.json')}")
    return model, metrics


if __name__ == '__main__':
    model, metrics = train_model(
        epochs=60,
        batch_size=32,
        lr=1e-3,
        save_path='models/rul_model.json'
    )
    print("\n✅ Training complete!")
