import torch
import torchaudio
import silero_vad

# Load the pre-trained VAD model
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')

# Extract necessary utility functions
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# Function to detect speech in an audio chunk
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

# Example usage
audio_path = r"C:\dev\bridge2bridge\audio\2024-09-27-11-12-08_large.wav"
waveform, sr = torchaudio.load(audio_path)

# Check if the audio chunk contains speech
speech_detected = is_speech(waveform, sr)
print(f"Speech detected: {speech_detected}")
