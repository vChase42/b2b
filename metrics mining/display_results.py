import pandas as pd
import matplotlib.pyplot as plt
import mplcursors

# Read the CSV data
small_data = pd.read_csv('output_data_diarize.csv')
large_data = pd.read_csv('output_data_large.csv')

# Sort both datasets by the 'duration' column
small_data = small_data.sort_values(by='duration')
large_data = large_data.sort_values(by='duration')

# Create the subplots for side-by-side plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot for small data
scatter1 = ax1.scatter(small_data['duration'], small_data['elapsed'], color='blue')
ax1.set_xlabel('Audio Clip Duration (s)')
ax1.set_ylabel('Transcription Time (s)')
ax1.set_title('Small Data: Transcription Time vs Audio Clip Duration')

# Plot for large data
scatter2 = ax2.scatter(large_data['duration'], large_data['elapsed'], color='green')
ax2.set_xlabel('Audio Clip Duration (s)')
ax2.set_ylabel('Transcription Time (s)')
ax2.set_title('Large Data: Transcription Time vs Audio Clip Duration')

# Adjust layout to prevent overlap
plt.tight_layout()

# Add hover functionality for small data
cursor1 = mplcursors.cursor(scatter1, hover=True)
cursor1.connect("add", lambda sel: sel.annotation.set_text(small_data['name'].iloc[sel.target.index]))

# Add hover functionality for large data
cursor2 = mplcursors.cursor(scatter2, hover=True)
cursor2.connect("add", lambda sel: sel.annotation.set_text(large_data['name'].iloc[sel.target.index]))

# Show the plot
plt.show()

# Calculate and print the average transcription time
average_small = sum(small_data['elapsed']) / len(small_data['elapsed'])
average_large = sum(large_data['elapsed']) / len(large_data['elapsed'])
print("Average transcription time (small data):", average_small)
print("Average transcription time (large data):", average_large)
