# instantiate the pipeline
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token="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")

# run the pipeline on an audio file
print("start")
diarization = pipeline("../audio/2024-10-02-16-23-51.wav")

# dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)


# print(dir(diarization))
# print("------------")
# print(vars(diarization))

for segment, track_info in diarization._tracks.items():
    start_time = segment.start
    end_time = segment.end
    duration = end_time - start_time
    speaker = list(track_info.values())[0]  # Extracting the speaker label
    print(f"Speaker: {speaker}, Start: {start_time:.2f}s, End: {end_time:.2f}s, Duration: {duration:.2f}s")


for segment in diarization.get_timeline():
    start_time = segment.start
    end_time = segment.end
    duration = end_time - start_time
    print(f"Start: {start_time:.2f}s, End: {end_time:.2f}s, Duration: {duration:.2f}s")


# for segment, track in diarization.itersegments():
#     start_time = segment.start
#     end_time = segment.end
#     duration = end_time - start_time
#     print(f"Start: {start_time:.2f}s, End: {end_time:.2f}s, Duration: {duration:.2f}s")

