from pyannote.audio import Pipeline
from pydub import AudioSegment
from pathlib import Path

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")

audio_file = "../audio/2024-10-02-16-23-51.wav"
file_name = Path(audio_file).name

diarization = pipeline(audio_file)
audio = AudioSegment.from_wav(audio_file)

counter = 0
filenames = []
for i, (segment, track_info) in enumerate(diarization._tracks.items()):
    start_time = segment.start * 1000  # Convert to milliseconds for pydub
    end_time = segment.end * 1000
    speaker = list(track_info.values())[0]  # Extract speaker label (e.g., 'SPEAKER_00')

    speaker_audio_segment = audio[start_time:end_time]

    output_filename = f"{file_name}_part{counter}.wav"
    filenames[i] = output_filename
    counter += 1

    speaker_audio_segment.export(output_filename, format="wav")
    print(f"Exported {output_filename} from {start_time/1000:.2f}s to {end_time/1000:.2f}s")
