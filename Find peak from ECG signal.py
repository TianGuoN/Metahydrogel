import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

file_path = 
data = pd.read_excel(file_path)

time = data.iloc[:, 0].values
ecg_signal = data.iloc[:, 1].values

R_peaks, _ = find_peaks(ecg_signal, distance= *, height= *)

S_peaks, _ = find_peaks(-ecg_signal, distance= *, height= *)

Q_peaks = []
for r in R_peaks:
    if r > *:  # 
        q_window = ecg_signal[r-*:r]  
        q_peak = np.argmin(q_window) + (r-*)  
        Q_peaks.append((time[q_peak], ecg_signal[q_peak]))

T_peaks = []
for r in R_peaks:
    if r < len(ecg_signal) - *:  
        t_window = ecg_signal[r+*:r+*]  
        t_peaks_in_window, _ = find_peaks(t_window, height= *) 
        if len(t_peaks_in_window) > 0:
            t_peak = t_peaks_in_window[0] + r + *  
            # Save timestamp and amplitude
            T_peaks.append((time[t_peak], ecg_signal[t_peak]))

plt.figure(figsize=(12, 6))
plt.plot(time, ecg_signal, label="ECG Signal")
plt.plot(time[R_peaks], ecg_signal[R_peaks], "x", label="R Peaks", color='r')
plt.plot(time[S_peaks], ecg_signal[S_peaks], "o", label="S Peaks", color='g')
plt.plot(time[Q_peaks], ecg_signal[Q_peaks], "v", label="Q Peaks", color='b')
plt.plot(time[T_peaks], ecg_signal[T_peaks], "^", label="T Peaks", color='m')
plt.title("ECG Signal with R, Q, S, and T Peaks")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()


peaks_df_list = []  
for r in R_peaks:
    peaks_df_list.append({'Peak Type': 'R', 'Time': time[r], 'Amplitude': ecg_signal[r]})
for s in S_peaks:
    peaks_df_list.append({'Peak Type': 'S', 'Time': time[s], 'Amplitude': ecg_signal[s]})
for q_time, q_amplitude in Q_peaks:
    peaks_df_list.append({'Peak Type': 'Q', 'Time': q_time, 'Amplitude': q_amplitude})


for t_time, t_amplitude in T_peaks:
    peaks_df_list.append({'Peak Type': 'T', 'Time': t_time, 'Amplitude': t_amplitude})

peaks_df = pd.DataFrame(peaks_df_list)

# Save
output_file_path =   
peaks_df.to_csv(output_file_path, index=False)

plt.plot(time, ecg_signal, label="ECG Signal")
plt.plot(time[R_peaks], ecg_signal[R_peaks], "x", label="R Peaks", color='r')

for q_time, q_amplitude in Q_peaks:
    plt.plot(q_time, q_amplitude, "v",  color='b')
# label="Q Peaks",
for t_time, t_amplitude in T_peaks:
    plt.plot(t_time, t_amplitude, "^", color='m')

plt.plot(time[S_peaks], ecg_signal[S_peaks], "o",  color='g')

plt.title("ECG Signal with R, Q, S, and T Peaks")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

 RR_intervals = np.diff(time[R_peaks])  
 QS_intervals = time[R_peaks] - time[S_peaks[:len(R_peaks)]]  

# Find min_length
 min_length = min(len(Q_peaks), len(T_peaks), len(R_peaks), len(S_peaks)) 
 Q_peaks = Q_peaks[:min_length]
 R_peaks = R_peaks[:min_length]
 S_peaks = S_peaks[:min_length]
 T_peaks = T_peaks[:min_length]
# QT_intervals
 QT_intervals = time[T_peaks] - time[Q_peaks]  

# DataFrame
 features_df = pd.DataFrame({
     'Q_peaks_time': time[Q_peaks],  
     'R_peaks_time': time[R_peaks],  
     'S_peaks_time': time[S_peaks],  
     'T_peaks_time': time[T_peaks],  
     'RR_intervals': RR_intervals[:min_length-1],  
     'QS_intervals': QS_intervals[:min_length],  
     'QT_intervals': QT_intervals  
 })
