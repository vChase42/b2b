from pyannote.audio import Pipeline
from pydub import AudioSegment
from pathlib import Path
from faster_whisper import WhisperModel
import torch
from dotenv import load_dotenv
import os
load_dotenv()
hf_key = os.getenv('HF_KEY')


pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_key)
whisper_model = WhisperModel("large-v3", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")

def diarize(diarization_pipeline, audio_file, output_folder, limit=1000):
    diarization = diarization_pipeline(audio_file)
    audio = AudioSegment.from_wav(audio_file)

    file_name = Path(audio_file).stem

    audio_segments = []
    speakers = []
    current_start_time = None
    current_end_time = None
    current_speaker = None

    for i, (segment, track_info) in enumerate(diarization._tracks.items()):
        start_time = segment.start * 1000
        end_time = segment.end * 1000
        speaker = list(track_info.values())[0]

        if speaker != current_speaker:
            if current_speaker != None:
                audio_segments.append((current_start_time, current_end_time))
                speakers.append(current_speaker)
            current_speaker = speaker
            current_start_time = start_time
            current_end_time = end_time
        else:
            if start_time - current_end_time > limit:
                print(f"large silence here from {current_end_time} to {start_time}")
            current_end_time = end_time

    # Add the last segment after the loop
    audio_segments.append((current_start_time, current_end_time))
    speakers.append(current_speaker)

    # Export the audio segments
    file_names = []
    for i, (start_time, end_time) in enumerate(audio_segments):
        speaker_audio_segment = audio[start_time:end_time]
        output_filename = f"{output_folder}/{file_name}_{speakers[i]}_part{i}.wav"
        speaker_audio_segment.export(output_filename, format="wav")
        print(f"Speaker {speakers[i]} spoke from {start_time/1000:.2f}s to {end_time/1000:.2f}s")
        file_names.append(output_filename)

    return speakers, file_names


def transcribe_audio(audio_path, model,pre_prompt):
    segments, info = model.transcribe(audio_path,initial_prompt = pre_prompt)
    text = " ".join([segment.text.strip() for segment in segments])   
    return text


if __name__ == "__main__":
    audio_file_path = "../audio/2024-10-02-16-23-51.wav"
    
    speakers, segmented_files = diarize(pipeline, audio_file_path, '.')
    
    full_text = []
    for segmented_file in segmented_files:
        transcription = transcribe_audio(segmented_file, whisper_model)
        print(f"Transcription for {segmented_file}:\n{transcription}\n")
        full_text.append(transcription)
    # output_text = " ".join(full_text)
    # print(output_text)

