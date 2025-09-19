import os
import scipy.io as sio
from scipy.io import loadmat
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch,decimate
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.svm import SVC
from gtda.time_series import TakensEmbedding
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, NumberOfPoints, Amplitude,Scaler
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed


def load_session_file(path):
    data = sio.loadmat(path)
    signal = data["signal"]
    fs = int(data["fs"].squeeze())
    timestamps = data["timestamps"].squeeze()
    channels = [ch[0] if isinstance(ch, np.ndarray) else ch for ch in data["channels"].squeeze()]
    return {
        "signal": signal,
        "fs": fs,
        "timestamps": timestamps,
        "channels": channels
    }

def load_all_sessions(root="session_wise_data"):
    dataset = {}
    for subj in sorted(os.listdir(root)):
        subj_path = os.path.join(root, subj)
        if not os.path.isdir(subj_path):
            continue
        sessions = {}
        for fname in sorted(os.listdir(subj_path)):
            if fname.endswith(".mat"):
                sess_id = os.path.splitext(fname)[0].split("_")[-1]  # session0, session1...
                sessions[sess_id] = load_session_file(os.path.join(subj_path, fname))
        dataset[subj] = sessions
    return dataset


all_data = load_all_sessions("session_wise_data")

print(f"Loaded {len(all_data)} subjects")
print(f"Subject S01 has {len(all_data['S01'])} sessions")
print("Shape of S01 session0:", all_data["S01"]["session0"]["signal"].shape)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def bandpass_filter(data, lowcut=0.1, highcut=50, fs=500, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data, axis=0)

def preprocess_session(session):
    sig = session["signal"]
    fs = session["fs"]

    sig_filt = bandpass_filter(sig, 1, 50, fs)
    return {
        "signal": sig_filt,
        "fs": fs,
        "timestamps": session["timestamps"],
        "channels": session["channels"]
    }

def preprocess_all(raw_data):
    processed = {}
    for subj, sessions in raw_data.items():
        processed[subj] = {}
        for sess_key, sess in sessions.items():
            processed[subj][sess_key] = preprocess_session(sess)
        print(f"Preprocessed {subj}")
    return processed


preprocessed_data = preprocess_all(all_data)

print("Shape:", preprocessed_data["S01"]["session0"]["signal"].shape)

def trim_sessions(all_data, target_len):
    
    trimmed = {}
    for subj, sessions in all_data.items():
        trimmed[subj] = {}
        for sess_id, sess in sessions.items():
            sig = sess["signal"]
            n = sig.shape[0]
            
            if n > target_len:
                trim = (n - target_len) // 2
                sig = sig[trim:trim+target_len, :]
                ts = sess["timestamps"][trim:trim+target_len]
            else:
                sig = sig
                ts = sess["timestamps"]
            
            trimmed[subj][sess_id] = {
                "signal": sig,
                "fs": sess["fs"],
                "timestamps": ts,
                "channels": sess["channels"]
            }
    return trimmed


data = trim_sessions(preprocessed_data,target_len=300000)

def downsample_session(session, target_fs=250):
    """Downsample one EEG session dict to target_fs Hz."""
    signal = session["signal"]
    fs = session["fs"]
    
    if fs == target_fs:
        return session  
    
    factor = fs // target_fs
    if fs % target_fs != 0:
        raise ValueError(f"Target fs={target_fs} not integer divisor of {fs}")
    
    sig_ds = decimate(signal, factor, axis=0, ftype='fir', zero_phase=True)
    ts_ds = session["timestamps"][::factor]
    
    return {
        "signal": sig_ds,
        "fs": target_fs,
        "timestamps": ts_ds,
        "channels": session["channels"]
    }

def downsample_all(trimmed_data, target_fs=250):
    ds_data = {}
    for subj, sessions in trimmed_data.items():
        ds_data[subj] = {}
        for sess_id, sess in sessions.items():
            ds_data[subj][sess_id] = downsample_session(sess, target_fs)
            print(f"{subj} {sess_id}: {sess['fs']} Hz → {target_fs} Hz, shape {ds_data[subj][sess_id]['signal'].shape}")
    return ds_data


ds_data = downsample_all(data, target_fs=250)

print("S01/session0 shape:", ds_data["S01"]["session0"]["signal"].shape)
print("Sampling rate:", ds_data["S01"]["session0"]["fs"])


window_size = 5000
step = 5000          
embed_d = 20
embed_tau = 1


labels_df = pd.read_csv("WAUC_datasets/subjective_ratings_with_labels.csv")

subject_labels = {}
for pid, group in labels_df.groupby("Participant ID"):
    subj_id = f"S{pid-1000:02d}"  
    mw_labels = group["mw_labels"].tolist()
    pw_labels = group["pw_labels"].tolist()
    subject_labels[subj_id] = {
        "mw": mw_labels,
        "pw": pw_labels,
    }

labeled_data = {}
for subj, sessions in ds_data.items():
    if subj not in subject_labels:
        print(f"Warning: {subj} not in CSV, skipping")
        continue
    
    labeled_data[subj] = {}
    for i, (sess_key, sess_data) in enumerate(sessions.items()):
        mw_label = subject_labels[subj]["mw"][i] if i < len(subject_labels[subj]["mw"]) else None
        pw_label = subject_labels[subj]["pw"][i] if i < len(subject_labels[subj]["pw"]) else None

        labeled_data[subj][sess_key] = {
            "signal": sess_data["signal"],
            "fs": sess_data["fs"],
            "timestamps": sess_data["timestamps"],
            "channels": sess_data["channels"],
            "mw_label": mw_label,
            "pw_label": pw_label,
        }

print("Example subject:", list(labeled_data.keys())[0])
print("Example sessions with labels:", labeled_data["S01"].keys())
print("Labels for S01/session1:", 
      "mw:", labeled_data["S21"]["session1"]["mw_label"], 
      "pw:", labeled_data["S21"]["session1"]["pw_label"]
      )


data = {}

for subj, sessions in labeled_data.items():
    subj_dict = {}
    for sess_key, sess_data in sessions.items():
        if sess_data["pw_label"] == 0:   
            subj_dict[sess_key] = {
                "signal": sess_data["signal"],
                "fs": sess_data["fs"],
                "label": sess_data["mw_label"]   
            }
    if subj_dict:  
        data[subj] = subj_dict

print("Subjects retained:", list(data.keys()))


all_windows = []
all_labels = []

for subj_key, sessions in data.items():   
    for sess_key, sess_data in sessions.items():
        sig = np.asarray(sess_data["signal"], dtype=float)
        label = sess_data["label"]   
        n_samples, n_channels = sig.shape

        for ch in range(n_channels):
            series = sig[:, ch]
            last_start = n_samples - window_size
            for start in range(0, last_start + 1, step):
                win = series[start:start + window_size]
                if np.isnan(win).any():
                    continue
                all_windows.append(win)
                all_labels.append(label)

X = np.stack(all_windows, axis=0)   
y = np.array(all_labels)            

print("X shape before embedding:", X.shape)
print("y shape (labels):", y.shape)

embedder = TakensEmbedding(dimension=embed_d, time_delay=embed_tau, stride=10)
X_emb = embedder.fit_transform(X)

print("Embedded shape:", X_emb.shape)


VR = VietorisRipsPersistence(homology_dimensions=[0, 1], n_jobs=-1)
diagrams = VR.fit_transform(X_emb)
print(diagrams.shape)


scaler = Scaler()
diagrams = scaler.fit_transform(diagrams)

pe = PersistenceEntropy()
nop = NumberOfPoints()
bottle = Amplitude(metric="bottleneck")
wasser = Amplitude(metric="wasserstein")

def extract_features(diagrams):
    """Extract features for one channel (all windows)."""
    f1 = pe.fit_transform(diagrams)          # shape (n_windows, 1)
    f2 = nop.fit_transform(diagrams)         # shape (n_windows, 1)
    f3 = bottle.fit_transform(diagrams)      # shape (n_windows, 1)
    f4 = wasser.fit_transform(diagrams)      # shape (n_windows, 1)
    
    feats = np.hstack([f1, f2, f3, f4])
    
    channel_vector = feats.flatten()
    return channel_vector

n_windows = 10
n_channels_total = diagrams.shape[0] // n_windows
print("Expecting", n_channels_total, "blocks")

X_features = []
y_labels = []

for ch in range(n_channels_total):
    start = ch * n_windows
    end = start + n_windows

    diagrams_block = diagrams[start:end]
    y_block = y[start:end]

    if len(y_block) != n_windows:
        print(f"Skipping block {ch}, wrong length {len(y_block)}")
        continue

    label = y_block[0]
    assert np.all(y_block == label), f"Mismatch in labels for channel {ch}"

    feats = extract_features(diagrams_block)
    X_features.append(feats)
    y_labels.append(label)

X_features = np.stack(X_features)
y_labels = np.array(y_labels)

print("Feature matrix:", X_features.shape)
print("Labels:", y_labels.shape)


y_labels = np.where(np.isin(y_labels, [0, 2, 4]), 0, 1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

best_params = {
    "C": 9.477495825843796,
    "degree": 4,
    "gamma": 0.15340913775402013,
    "kernel": "rbf",
}

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(**best_params, probability=True, random_state=42)),
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print("Final model performance with best params:")
print("Accuracy:", acc)
print("Balanced Accuracy:", bal_acc)
print("F1-macro:", f1)

