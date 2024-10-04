from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/speafsefker-segmentation",use_auth_token="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")

audio_file = "./audio/2024-10-02-16-23-51.wav"

speaker_change = pipeline(audio_file)

for speech_turn, _, speaker in speaker_change.itertracks(yield_label=True):
    start_time = speech_turn.start
    end_time = speech_turn.end
    print(f"Speaker change detected from {start_time:.2f}s to {end_time:.2f}s")

