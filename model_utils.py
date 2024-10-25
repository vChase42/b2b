import wave
from pyannote.audio import Pipeline
from pydub import AudioSegment
from pathlib import Path
from faster_whisper import WhisperModel
import torch
from dotenv import load_dotenv
import os
import torchaudio
import numpy as np
load_dotenv()
hf_key = os.getenv('HF_KEY')


class DiarizationManager:
    def __init__(self, pipeline):
        #a set of audio chunks that contain unique voice signatures'
        self.pipeline = pipeline
        self.speaker_signatures = []
        self.speaker_names = []

    #THIS PROBLEM IS GETTING HARD.
    #maybe the only signatures i should save should be high quality ones.

    def diarize(self, audio_file, output_folder):
        file_name = Path(audio_file).stem

        combined_audio_segment = self._prepare_combined_audio(audio_file)
        combined_audio_file = self._export_combined_audio(combined_audio_segment, output_folder, file_name)

        total_signature_duration_ms = sum(len(signature) + 1000 for signature in self.speaker_signatures)
        total_signature_duration = total_signature_duration_ms / 1000.0

        # Perform diarization
        diarization = self.pipeline(combined_audio_file)
        pre_audio = AudioSegment.from_wav(audio_file)
        tuples = []

        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if segment.end <= total_signature_duration:
                continue

            adjusted_start = segment.start - total_signature_duration
            adjusted_end = segment.end - total_signature_duration

            adjusted_start_ms = max(adjusted_start * 1000, 0)
            adjusted_end_ms = min(adjusted_end * 1000, len(pre_audio))

            speaker_audio_segment = pre_audio[adjusted_start_ms:adjusted_end_ms]

            output_filename = f"{output_folder}/{file_name}_{speaker}_part{int(segment.start)}.wav"
            speaker_audio_segment.export(output_filename, format="wav")
            print(f"Speaker {speaker} spoke from {adjusted_start:.2f}s to {adjusted_end:.2f}s")

            if speaker not in self.speaker_names:
                self.speaker_signatures.append(speaker_audio_segment)
                self.speaker_names.append(speaker)

            tuples.append({
                'speaker': speaker,
                'audiofile': output_filename,
                'start_seconds': max(adjusted_start - 0.5, 0),
                'end_seconds': min(adjusted_end + 0.5, len(pre_audio) / 1000.0)
            })

        return tuples

    def _prepare_combined_audio(self, audio_file):
        pre_audio = AudioSegment.from_wav(audio_file)
        one_second_silence = AudioSegment.silent(duration=1000)
        combined_audio_segment = AudioSegment.empty()

        for signature in self.speaker_signatures:
            combined_audio_segment += signature + one_second_silence

        combined_audio_segment += pre_audio

        return combined_audio_segment

    def _export_combined_audio(self, combined_audio_segment, output_folder, file_name):
        output_file_name = f"{file_name}_signatures.wav"
        combined_audio_file = os.path.join(output_folder, output_file_name)
        combined_audio_segment.export(combined_audio_file, format="wav")
        return combined_audio_file

    @staticmethod
    def print_dict(tuple):
        print(f"Speaker: {tuple['speaker']}, Start Time: {tuple['start_seconds']}, End Time: {tuple['end_seconds']}, File Name: {tuple['audiofile']}")

#audio is split everytime there is a 3 seconds silence, or a transition between current speakers.
def diarize(diarization_pipeline, audio_file, output_folder, limit=3000):
    diarization = diarization_pipeline(audio_file)

    audio = AudioSegment.from_wav(audio_file)

    file_name = Path(audio_file).stem

    audio_segments = []
    speakers = []
    current_start_time = None
    current_end_time = None
    current_speaker = None

    #aggregate the data
    for i, (segment, track_info) in enumerate(diarization._tracks.items()):
        start_time = segment.start * 1000
        end_time = segment.end * 1000
        speaker = list(track_info.values())[0]

        if speaker != current_speaker or start_time - current_end_time > 1000:
            if current_speaker != None:
                audio_segments.append((max(current_start_time-500,0), min(current_end_time+500,len(audio))))
                speakers.append(current_speaker)
            current_speaker = speaker
            current_start_time = start_time
            current_end_time = end_time
        else:
            if start_time - current_end_time > limit:
                print(f"large silence here from {1.0*current_end_time/1000} to {1.0*start_time/1000}")
            current_end_time = end_time
    # Add the last segment after the loop
    if current_start_time is not None:
        audio_segments.append((current_start_time, current_end_time))
        speakers.append(current_speaker)


    # prepare audio segments for export
    tuples = []
    for i, (start_time, end_time) in enumerate(audio_segments):
        speaker_audio_segment = audio[start_time:end_time]
        output_filename = f"{output_folder}/{file_name}_{speakers[i]}_part{i}.wav"
        speaker_audio_segment.export(output_filename, format="wav")
        print(f"Speaker {speakers[i]} spoke from {start_time/1000:.2f}s to {end_time/1000:.2f}s")

        tuples.append({
            'speaker':speakers[i],
            'audiofile':output_filename,
            'start_seconds':1.0*start_time/1000,
            'end_seconds':1.0*end_time/1000
        })



    return tuples


def transcribe_audio(audio_path, model,pre_prompt):
    segments, info = model.transcribe(audio_path,initial_prompt = pre_prompt)
    text = " ".join([segment.text.strip() for segment in segments])   
    return text


def is_speech(vad_model, get_speech_timestamps, audio_file):
    waveform, sample_rate = torchaudio.load(audio_file)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000  # Ensure the sample rate is set to 16000 after resampling
    
    if len(waveform.shape) > 1:    # Convert stereo to mono if necessary
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    speech_timestamps = get_speech_timestamps(waveform, vad_model, sampling_rate=sample_rate)
    return len(speech_timestamps) > 0  # True if any speech detected, False otherwise



if __name__ == "__main__":
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_key)
    whisper_model = WhisperModel("large-v3", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")

    audio_file_path = "../audio/2024-10-02-16-23-51.wav"
    
    speakers, segmented_files = diarize(pipeline, audio_file_path, '.')
    
    full_text = []
    for segmented_file in segmented_files:
        transcription = transcribe_audio(segmented_file, whisper_model)
        print(f"Transcription for {segmented_file}:\n{transcription}\n")
        full_text.append(transcription)
    # output_text = " ".join(full_text)
    # print(output_text)

