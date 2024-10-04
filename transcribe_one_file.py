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
    model = whisperx.load_model(model_name, device)

    # Check if file exists
    if not os.path.exists(file_name):
        print(f"File {file_name} does not exist.")
        return

    # Read audio data
    audio_data, sample_rate = sf.read(file_name)

    # Transcribe using WhisperX
    result = model.transcribe(file_name)

    # Display transcription results
    segments = result['segments']
    transcription = ' '.join([segment['text'] for segment in segments])
    print(f"Transcription: {transcription}")

def get_file_size(file_name):
    if not os.path.exists(file_name): return -1
    return os.path.getsize(file_name) / 1024

file_name = "./audio/clip1_fixed.wav"
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    size = get_file_size(file_name)
    duration = get_wav_duration(file_name)
    model_name = "small"

    start_time = time.time()
    transcribe_audio(file_name, model_name)

    elapsed_time = time.time() - start_time
    print("----------------------\n\n\n")
    print("Model used:",model_name, "File Name:",file_name)
    print("File size:",size,"KB")
    print("Audio Duration:",duration)
    print(f"Time taken to transcribe text: {elapsed_time}")
