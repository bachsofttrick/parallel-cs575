import numpy as np
import matplotlib.pyplot as plt

# Define the frequency of the sine wave
frequency = 1  # Hz

# Calculate the period
period = 1 / frequency

# Define the x-values for one period
t = np.linspace(0, 2 * np.pi, 1000)

# Calculate the corresponding y-values (sine wave)
y = np.sin(2 * np.pi * frequency * t)

# Plot the sine wave
plt.plot(t, y)
plt.title('Harmonic Sine Wave - Two Periods')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()