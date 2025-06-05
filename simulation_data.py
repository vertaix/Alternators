import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

##################### Create the states
# Parameters
fs = 500  # Sampling frequency
t = np.arange(.25, 1, 1 / fs)  # Time vector from 0 to 2 seconds
low_freq = 2  # Low frequency in Hz
high_freq1 = 60  # High frequency for the first window in Hz
high_freq2 = 60  # High frequency for the second window in Hz

# Generate base low-frequency signal (sine wave)
base_signal = np.sin(2 * np.pi * low_freq * t)

# Generate high-frequency signals for the modulation windows
high_freq_signal1 =2* np.sin(2 * np.pi * high_freq1 * t)
high_freq_signal2 = 2*np.sin(2 * np.pi * high_freq2 * t)

# Define modulation windows (in seconds)
modulation_windows = [(0.5, 0.9), (1.2, 1.5)]  # Example windows

# Create a copy of the base signal for modulation
modulated_signal = base_signal.copy()

# Apply modulation in the specified windows
for i, (start, end) in enumerate(modulation_windows):
    start_idx = int(start * fs)
    end_idx = int(end * fs)
    if i == 0:
        modulated_signal[start_idx:end_idx] += high_freq_signal1[start_idx:end_idx]
    elif i == 1:
        modulated_signal[start_idx:end_idx] += high_freq_signal2[start_idx:end_idx]
modulated_signal=modulated_signal+np.random.randn(modulated_signal.shape[0])*.2

# Plot the signals
plt.figure(figsize=(14, 6))

# Plot the modulated signal
plt.subplot(3, 1, 1)
plt.plot(t, modulated_signal, label='Modulated Signal', color='r')
plt.title('Latents (Z)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()

############### Create the observations

# Parameters for noise
num_observations = 10
mean = np.zeros_like(modulated_signal)  # Mean of the multivariate normal distribution
# cov = np.identity(len(modulated_signal)) * 0.2  # Covariance matrix for the multivariate normal distribution

# Create noisy observations
noisy_observations = []
for _ in range(num_observations):
    scale_factor = np.random.uniform(0.5, 1.5)  # Randomly scale y
    cov_scale = np.random.uniform(0.3, .8)  # Randomly scale the covariance
    cov = np.identity(len(modulated_signal)) * cov_scale
    scaled_y = modulated_signal * scale_factor
    noise = np.random.multivariate_normal(mean, cov)
    noisy_observation = scaled_y + noise
    noisy_observations.append(noisy_observation)

# Convert to numpy array for easier handling
noisy_observations = np.array(noisy_observations)

# Plot the base trajectory and noisy observations
plt.subplot(3, 1, 2)
for i, obs in enumerate(noisy_observations):
    plt.plot(t,obs+i*modulated_signal.max(), label=f'Noisy Observation {i + 1}', alpha=0.6)
plt.title('Observations (X)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.tight_layout()
############## Vendi
from vendi import score
from sklearn.metrics.pairwise import cosine_similarity

L = 100
# Compute similarities
def mse(x,y):
    return (x-y)**2
vendi_scores = []
for i in range(1, len(noisy_observations[0]) - L + 1):
    # prev_window = noisy_observations[:, i - 1:i - 1 + L]
    # curr_window = noisy_observations[:, i:i + L]

    # prev_window = modulated_signal[ i - 1:i - 1 + L]
    curr_window = modulated_signal[ i:i + L]

    # vendi diversity
    scores = score(curr_window.reshape([-1, 1]),mse, q=.1)
    vendi_scores.append(scores)


vendi_scores=np.array(vendi_scores)
# Plot the sivendi_scoresmilarities
plt.subplot(3, 1, 3)
plt.plot(t[ L:], vendi_scores, label='vendi_scores between windows')
plt.title('Vendi score between Consecutive Windows of Noisy Observations (X)')
plt.xlabel('Time Step')
plt.ylabel('Vendi Score')
plt.tight_layout()
plt.show()

dataset= {'xs': noisy_observations.T[L:]/noisy_observations.max(axis=1),
             'vendis': np.expand_dims(vendi_scores, axis=-1)/vendi_scores.max(),
             'latents': np.expand_dims(modulated_signal[L:], axis=-1)/modulated_signal.max()}

import pickle
pickle.dump(dataset, open("dataset_simple.p", "wb"))