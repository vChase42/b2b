import wave
import os

def get_file_size(file_name):
    if not os.path.exists(file_name): return -1
    return os.path.getsize(file_name) / 1024
def get_wav_duration(file_name):
    with wave.open(file_name, 'r') as wav_file:
        # Extract parameters from the .wav file
        frame_rate = wav_file.getframerate()  # Frames per second
        n_frames = wav_file.getnframes()  # Total number of frames
        duration = n_frames / float(frame_rate)  # Duration in seconds
    return duration



names_list = []
sizes = []
durations = []

with open("clips2.txt") as names:
    for n in names:
        file_name = "./audio/" + n.strip()

        size = get_file_size(file_name)
        duration = get_wav_duration(file_name)

        print(f"{file_name}, File Size: {size} KB, Audio Duration: {duration}")
        names_list.append(file_name)
        sizes.append(size)
        durations.append(duration)

import matplotlib.pyplot as plt
# Sorting the sizes and durations while keeping the filenames in sync
sorted_sizes = sorted(sizes)
sorted_durations = sorted(durations)

# Create numeric indices for the bars (1, 2, 3, ...)
indices_sizes = range(1, len(sorted_sizes) + 1)
indices_durations = range(1, len(sorted_durations) + 1)

# Create subplots: 1 row, 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plotting the sorted bar chart for file sizes
ax1.barh(indices_sizes, sorted_sizes, color='skyblue')
ax1.set_xlabel('File Size (KB)')
ax1.set_ylabel('Audio Files (Index)')
ax1.set_title('Sorted Bar Chart of File Sizes')
ax1.set_yticks([])  # Remove y-axis labels (no names)

# Plotting the sorted bar chart for audio durations
ax2.barh(indices_durations, sorted_durations, color='lightgreen')
ax2.set_xlabel('Duration (seconds)')
ax2.set_ylabel('Audio Files (Index)')
ax2.set_title('Sorted Bar Chart of Audio Durations')
ax2.set_yticks([])  # Remove y-axis labels (no names)

# Adjust layout
plt.tight_layout()
plt.show()


with open("output_data","w") as outer:
    for i in range(len(names_list)):
        new_line = f"{names_list[i]}, {sizes[i]:10.2f},{durations[i]:10.2f}\n"
        outer.write(new_line)