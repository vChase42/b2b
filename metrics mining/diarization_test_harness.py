import time
import wave
from dotenv import load_dotenv
import os
load_dotenv()
hf_key = os.getenv('HF_KEY')

from pyannote.audio import Pipeline
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token=hf_key)

pipeline.to(device)

def get_wav_duration(file_name):
    with wave.open(file_name, 'r') as wav_file:
        # Extract parameters from the .wav file
        frame_rate = wav_file.getframerate()  # Frames per second
        n_frames = wav_file.getnframes()  # Total number of frames
        duration = n_frames / float(frame_rate)  # Duration in seconds
    return duration


if __name__ == "__main__":
    transcription_times = []
    names_list = []
    durations = []
    with open("clips2.txt") as names:
        for n in names:
            file_name = "../audio/" + n.strip()
            duration = get_wav_duration(file_name)
            

            start_time = time.time()
            diarization = pipeline(file_name)
            elapsed_time = time.time() - start_time
            print("Elapsed time:",elapsed_time)
            transcription_times.append(elapsed_time)
            names_list.append(file_name)
            durations.append(duration)
    


    with open("output_data","w") as outer:
        outer.write("name,duration,elapsed\n")
        for i in range(len(names_list)):

            new_line = f"{names_list[i]}, {durations[i]:10.2f},{transcription_times[i]:10.4}\n"
            print(new_line)
            outer.write(new_line)

