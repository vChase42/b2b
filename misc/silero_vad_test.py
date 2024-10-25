import torch
import torchaudio
import silero_vad

vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

def is_speech(vad_model, get_speech_timestamps, audio_chunk, sample_rate):
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_chunk = torch.from_numpy(audio_chunk).float()
        audio_chunk = resampler(audio_chunk)
        sample_rate = 16000  # Ensure the sample rate is set to 16000 after resampling
    
    if len(audio_chunk.shape) > 1:    # Convert stereo to mono if necessary
        audio_chunk = torch.mean(audio_chunk, dim=0, keepdim=True)

    speech_timestamps = get_speech_timestamps(audio_chunk, vad_model, sampling_rate=sample_rate)
    return len(speech_timestamps) > 0  # True if any speech detected, False otherwise


if __name__ == "__main__":
    audio_path = r"C:\dev\bridge2bridge\audio\2024-09-27-11-12-08_large.wav"
    waveform, sr = torchaudio.load(audio_path)

    # Check if the audio chunk contains speech
    speech_detected = is_speech(vad_model, get_speech_timestamps, waveform, sr)
    print(f"Speech detected: {speech_detected}")
