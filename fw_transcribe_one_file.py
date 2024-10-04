from faster_whisper import WhisperModel
import torch

def transcribe_audio(audio_path, model_size):
    model = WhisperModel(model_size, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")

    segments, info = model.transcribe(audio_path)

    language = info[0]   # Language of the transcription
    duration = info[1]   # Duration of the audio file

    text = " ".join([segment.text.strip() for segment in segments])
    return text
    


if __name__ == "__main__":
    audio_file_path = "./audio/2024-10-02-16-23-51.wav"
    text = transcribe_audio(audio_file_path, model_size="large-v3")
    print(text)