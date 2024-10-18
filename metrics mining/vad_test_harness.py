import torch
import torchaudio
import silero_vad
import time
import wave

# Load the pre-trained VAD model
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')

# Extract necessary utility functions
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils


def get_wav_duration(file_name):
    with wave.open(file_name, 'r') as wav_file:
        # Extract parameters from the .wav file
        frame_rate = wav_file.getframerate()  # Frames per second
        n_frames = wav_file.getnframes()  # Total number of frames
        duration = n_frames / float(frame_rate)  # Duration in seconds
    return duration

def is_speech(audio_chunk, sample_rate):
    # Resample audio to 16kHz if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_chunk = resampler(audio_chunk)
        sample_rate = 16000  # Ensure the sample rate is set to 16000 after resampling
    
    # Convert stereo to mono if necessary
    if len(audio_chunk.shape) > 1:
        audio_chunk = torch.mean(audio_chunk, dim=0, keepdim=True)  # Average the channels to mono

    # Use the get_speech_timestamps utility function
    speech_timestamps = get_speech_timestamps(audio_chunk, vad_model, sampling_rate=sample_rate)
    return len(speech_timestamps) > 0  # True if any speech detected, False otherwise



if __name__ == "__main__":
    transcription_times = []
    names_list = []
    durations = []
    with open("clips2.txt") as names:
        for n in names:
            file_name = "../audio/" + n.strip()
            duration = get_wav_duration(file_name)
            waveform, sr = torchaudio.load(file_name)

            start_time = time.time()
            speech_flag = is_speech(waveform,sr)
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

