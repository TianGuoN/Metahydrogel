import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from pyts.image import RecurrencePlot

file_path = 
data = pd.read_excel(file_path)

time = data.iloc[:, 0].values
ecg_signal = data.iloc[:, 1].values

peaks, _ = find_peaks(ecg_signal, height=**, distance=**)  
peak_times = time[peaks]
peak_values = ecg_signal[peaks]

RR_intervals = np.diff(peak_times)

plt.figure(figsize=(10, 8))
plt.scatter(RR_intervals[:-1], RR_intervals[1:],  edgecolor='b', facecolor='none')
plt.xlabel('RR_n (s)')
plt.ylabel('RR_{n+1} (s)')
plt.title('Poincaré Plot of RR Intervals')
plt.grid(True)
plt.tight_layout()
plt.show()

heart_rate = 60 / RR_intervals

scaler = StandardScaler()
heart_rate_scaled = scaler.fit_transform(heart_rate.reshape(-1, 1)).flatten()

rp = RecurrencePlot(threshold=**, percentage=**)  
X_rp = rp.fit_transform(heart_rate_scaled.reshape(1, -1))

plt.figure(figsize=(10, 10))
plt.imshow(X_rp[0], cmap='gray_r', origin='lower') 
plt.title('Recurrent Plot of Heart Rate')
plt.xlabel('Time')
plt.ylabel('Time')
plt.colorbar(label='Recurrence Intensity')
plt.grid(False)
plt.tight_layout()
plt.show()

def compute_rqa(X_rp):
    N = X_rp.shape[0]
    RR = np.sum(X_rp) / (N * N)

    diagonal_counts = []
    max_diag_len = 0
    current_diag_len = 0

    for k in range(-N + 1, N): 
        diag = np.diag(X_rp, k=k)
        diag_len = 0
        for i in range(len(diag)):
            if diag[i] == 1:
                diag_len += 1
            else:
                if diag_len > 0:
                    diagonal_counts.append(diag_len)
                    max_diag_len = max(max_diag_len, diag_len)
                diag_len = 0
        if diag_len > 0: 
            diagonal_counts.append(diag_len)
            max_diag_len = max(max_diag_len, diag_len)

    diagonal_counts = np.array(diagonal_counts)
    DET = np.sum(diagonal_counts[diagonal_counts >= 2]) / np.sum(diagonal_counts)
    L = np.mean(diagonal_counts[diagonal_counts >= 2])
    Lmax = max_diag_len
    ENT = -np.sum((diagonal_counts[diagonal_counts >= 2] / np.sum(diagonal_counts)) *
                  np.log(diagonal_counts[diagonal_counts >= 2] / np.sum(diagonal_counts)))

    return {'RR': RR, 'DET': DET, 'L': L, 'Lmax': Lmax, 'ENT': ENT}

rqa_results = compute_rqa(X_rp[0])

window_size = **  
step_size = **    

rr_time_indices = []
rr_values = []
det_values = []

for start in range(0, len(heart_rate_scaled) - window_size + 1, step_size):
    end = start + window_size
    segment = heart_rate_scaled[start:end].reshape(1, -1)

    rp_segment = rp.fit_transform(segment)[0]

    rqa_metrics = compute_rqa(rp_segment)

    rr_values.append(rqa_metrics['RR'])
    det_values.append(rqa_metrics['DET'])

    segment_time = np.mean(peak_times[start:end])
    rr_time_indices.append(segment_time)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(rr_time_indices, rr_values, marker='o', linestyle='-', color='purple')
plt.ylabel('Recurrence Rate (RR)')
plt.title('Sliding Window Recurrence Analysis')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(rr_time_indices, det_values, marker='s', linestyle='-', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Determinism (DET)')
plt.grid(True)

plt.tight_layout()
plt.show()

df_result = pd.DataFrame({
    'Time (s)': rr_time_indices,
    'Recurrence Rate (RR)': rr_values,
    'Determinism (DET)': det_values
})
df_result.to_excel('rr_det_over_time.xlsx', index=False)