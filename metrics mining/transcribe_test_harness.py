import os
import whisperx
import soundfile as sf
import torch
import warnings
import logging
import time
import wave
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def get_wav_duration(file_name):
    with wave.open(file_name, 'r') as wav_file:
        # Extract parameters from the .wav file
        frame_rate = wav_file.getframerate()  # Frames per second
        n_frames = wav_file.getnframes()  # Total number of frames
        duration = n_frames / float(frame_rate)  # Duration in seconds
    return duration

# Constants

def transcribe_audio(file_name, model_name):
    # Load WhisperX model
    global model

    # Check if file exists
    if not os.path.exists(file_name):
        print(f"File {file_name} does not exist.")
        return


    
    # Transcribe using WhisperX
    start_time = time.time()
    result = model.transcribe(file_name)
    elapsed_time = time.time() - start_time
    print(elapsed_time)

    # Display transcription results
    segments = result['segments']
    transcription = ' '.join([segment['text'] for segment in segments])
    print(f"Transcription: {transcription}")
    return elapsed_time

def get_file_size(file_name):
    if not os.path.exists(file_name): return -1
    return os.path.getsize(file_name) / 1024

device = "cuda" if torch.cuda.is_available() else "cpu"

names_list = []
sizes = []
durations = []
transcription_times = []

model_name = "small"

model = whisperx.load_model(model_name, device)

if __name__ == "__main__":
    with open("clips2.txt") as names:
        for n in names:
            file_name = "../audio/" + n.strip()

            size = get_file_size(file_name)
            duration = get_wav_duration(file_name)

            elapsed_time = transcribe_audio(file_name, model_name)
            
            print(f"{file_name}, File Size: {size} KB, Audio Duration: {duration}, Elapsed Time: {elapsed_time}")
            names_list.append(file_name)
            sizes.append(size)
            durations.append(duration)
            transcription_times.append(elapsed_time)



with open("output_data","w") as outer:
    for i in range(len(names_list)):
        new_line = f"{names_list[i]}, {sizes[i]:10.2f},{durations[i]:10.2f},{transcription_times[i]:10.4}\n"
        outer.write(new_line)
