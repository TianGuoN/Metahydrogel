import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 读取ECG信号数据
file_path = r'D:\ML\P16 Fatigue state detection\ECG05.xlsx'  # 请将此路径替换为你的实际文件路径
data = pd.read_excel(file_path)

# 假设第一列是时间，第二列是ECG信号
time = data.iloc[:, 0].values
ecg_signal = data.iloc[:, 1].values

# 使用find_peaks检测R波（我们假设R波是ECG中最高的正峰值）
R_peaks, _ = find_peaks(ecg_signal, distance=150, height=0.42)

# S波是R波之后的低谷
S_peaks, _ = find_peaks(-ecg_signal, distance=150, height=0.2)

# 提取Q波：R波之前的最小值
Q_peaks = []
for r in R_peaks:
    if r > 50:  # 确保有足够的空间向前寻找Q波
        q_window = ecg_signal[r-50:r]  # 在R波之前的50个数据点内搜索
        q_peak = np.argmin(q_window) + (r-50)  # 找到最小值的索引
        # 存储时间和幅值
        Q_peaks.append((time[q_peak], ecg_signal[q_peak]))

# 提取T波：R波之后的第一个显著正峰值
T_peaks = []
for r in R_peaks:
    if r < len(ecg_signal) - 100:  # 确保有足够的空间向后寻找T波
        # 在R波之后的100个数据点内寻找峰值
        t_window = ecg_signal[r+20:r+400]  # 排除R波附近的区域，从R波之后20个点开始
        t_peaks_in_window, _ = find_peaks(t_window, height=0.2)  # 寻找正峰值
        if len(t_peaks_in_window) > 0:
            t_peak = t_peaks_in_window[0] + r + 20  # 找到的T波峰值相对原始信号的索引
            # 存储时间和幅值
            T_peaks.append((time[t_peak], ecg_signal[t_peak]))

# # 将提取的特征峰绘制出来
# plt.figure(figsize=(12, 6))
# plt.plot(time, ecg_signal, label="ECG Signal")
# plt.plot(time[R_peaks], ecg_signal[R_peaks], "x", label="R Peaks", color='r')
# plt.plot(time[S_peaks], ecg_signal[S_peaks], "o", label="S Peaks", color='g')
# plt.plot(time[Q_peaks], ecg_signal[Q_peaks], "v", label="Q Peaks", color='b')
# plt.plot(time[T_peaks], ecg_signal[T_peaks], "^", label="T Peaks", color='m')
# plt.title("ECG Signal with R, Q, S, and T Peaks")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.show()

# 创建包含所有峰值的DataFrame
peaks_df_list = []  # 使用列表来收集数据

# 添加R波
for r in R_peaks:
    peaks_df_list.append({'Peak Type': 'R', 'Time': time[r], 'Amplitude': ecg_signal[r]})

# 添加S波
for s in S_peaks:
    peaks_df_list.append({'Peak Type': 'S', 'Time': time[s], 'Amplitude': ecg_signal[s]})

# 添加Q波
for q_time, q_amplitude in Q_peaks:
    peaks_df_list.append({'Peak Type': 'Q', 'Time': q_time, 'Amplitude': q_amplitude})

# 添加T波
for t_time, t_amplitude in T_peaks:
    peaks_df_list.append({'Peak Type': 'T', 'Time': t_time, 'Amplitude': t_amplitude})

# 创建 DataFrame
peaks_df = pd.DataFrame(peaks_df_list)

# 保存到CSV文件
output_file_path = r'D:\ML\P16 Fatigue state detection\ECG_peaks.csv'  # 请将此路径替换为您想要保存的路径
peaks_df.to_csv(output_file_path, index=False)

print("Peak values saved to", output_file_path)

# 将提取的特征峰绘制出来
plt.plot(time, ecg_signal, label="ECG Signal")
plt.plot(time[R_peaks], ecg_signal[R_peaks], "x", label="R Peaks", color='r')

# 绘制S、Q和T波的峰值
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

print("Peak values saved to", output_file_path)

# # 提取各峰的间隔
# RR_intervals = np.diff(time[R_peaks])  # R波之间的间隔
# QS_intervals = time[R_peaks] - time[S_peaks[:len(R_peaks)]]  # QRS区间时间
#
# # 找到最小长度
# min_length = min(len(Q_peaks), len(T_peaks), len(R_peaks), len(S_peaks))  # 确保四个列表长度一致
#
# # 截取到最小长度
# Q_peaks = Q_peaks[:min_length]
# R_peaks = R_peaks[:min_length]
# S_peaks = S_peaks[:min_length]
# T_peaks = T_peaks[:min_length]
#
# # 计算QT区间
# QT_intervals = time[T_peaks] - time[Q_peaks]  # QT区间时间
#
# # 创建一个包含这些参数的DataFrame
# features_df = pd.DataFrame({
#     'Q_peaks_time': time[Q_peaks],  # 确保长度一致
#     'R_peaks_time': time[R_peaks],  # 确保长度一致
#     'S_peaks_time': time[S_peaks],  # 确保长度一致
#     'T_peaks_time': time[T_peaks],  # 确保长度一致
#     'RR_intervals': RR_intervals[:min_length-1],  # RR间隔的长度比R波少1
#     'QS_intervals': QS_intervals[:min_length],  # QS间隔的长度
#     'QT_intervals': QT_intervals  # QT间隔
# })
#
# print(features_df)