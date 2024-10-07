from dotenv import load_dotenv
import os
load_dotenv()
hf_key = os.getenv('HF_KEY')

import json
import datetime
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np
import gradio as gr
import soundfile as sf
import webrtcvad
from faster_whisper import WhisperModel
import torch

from split_and_transcribe import diarize, transcribe_audio
from dialog_manager import DialogManager
from pyannote.audio import Pipeline


class TranscriptionProcessor:
    def __init__(self):
        # Initialize variables
        self.pre_prompt_file = 'pre-prompt.json'
        self.whisper_language = 'english'
        self.silence_duration_limit = 1.0  # Seconds
        self.max_buffer_duration = 10  # Seconds

        # VAD and audio processing
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)  # Aggressiveness mode (0-3)
        self.silence_duration = 0.0  # Accumulated silence duration
        self.sample_rate = 16000  # VAD works with 16kHz audio
        self.frame_duration_ms = 30  # Frame size for VAD

        # Load pre-prompt words
        self.load_pre_prompt_from_file()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_key)
        self.pipeline.to(device)
        self.model_large = WhisperModel("large-v3", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
        self.model_small = WhisperModel("small", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")



        self.audio_buffer = np.array([], dtype=np.float32)
        self.current_buffer_start_time = None
        self.pre_prompt_words = []

        self.dialog_manager = DialogManager()

        #lock
        self.lock = Lock()

    # Load pre-prompt words from file
    def load_pre_prompt_from_file(self):
        if os.path.exists(self.pre_prompt_file):
            with open(self.pre_prompt_file, 'r') as file:
                self.pre_prompt_words = json.load(file)

    # Save pre-prompt words to file
    def save_pre_prompt_to_file(self):
        with open(self.pre_prompt_file, 'w') as file:
            json.dump(self.pre_prompt_words, file)

    # Function to detect silence using VAD
    # def detect_silence(self, audio_chunk, sr):
    #     # Resample to 16kHz if necessary
    #     if sr != self.sample_rate:
    #         number_of_samples = round(len(audio_chunk) * float(self.sample_rate) / sr)
    #         audio_chunk = scipy.signal.resample(audio_chunk, number_of_samples)
    #         sr = self.sample_rate

    #     # Convert to 16-bit PCM
    #     max_int16 = np.iinfo(np.int16).max
    #     audio_int16 = (audio_chunk * max_int16).astype(np.int16)

    #     # Split audio into frames
    #     frame_size = int(sr * self.frame_duration_ms / 1000)
    #     frames = [
    #         audio_int16[i:i + frame_size]
    #         for i in range(0, len(audio_int16), frame_size)
    #     ]

    #     is_silence_detected = False
    #     for frame in frames:
    #         if len(frame) < frame_size:
    #             continue  # Skip incomplete frames

    #         frame_bytes = frame.tobytes()
    #         is_speech = True
    #         # is_speech = self.vad.is_speech(frame_bytes, sr)

    #         if not is_speech:
    #             self.silence_duration += self.frame_duration_ms / 1000.0
    #         else:
    #             self.silence_duration = 0.0  # Reset if speech is detected

    #         if self.silence_duration >= self.silence_duration_limit:
    #             is_silence_detected = True
    #             break  # No need to process further frames

    #     return is_silence_detected

    def format_audio_stream(self,y):
        # Convert the audio to the correct format
        
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        return y


    # Function to process audio buffer with the specified model
    def process_audio_buffer(self, model, sample_rate, clearBuffer):
        with self.lock:
            audio_to_process = self.audio_buffer.copy()

        if len(audio_to_process) == 0:
            return  # Nothing to process

        # Save audio to file
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")       
        audio_file = f"./audio/{timestamp}.wav"
        os.makedirs('./audio', exist_ok=True)
        sf.write(audio_file, self.format_audio_stream(audio_to_process), sample_rate)
        transcribed_text = ""
        
        
        #split audio by speaker
        output_folder = "./audio_segments"
        os.makedirs(output_folder, exist_ok=True)
        speakers, filenames, times = diarize(self.pipeline, audio_file, output_folder)

        try:
            for i, audio_file in enumerate(filenames):
                initial_prompt = f"{' '.join(self.pre_prompt_words)}"   #need to bring add previously generated text here.
                text = transcribe_audio(audio_file, model, initial_prompt)

                transcribed_text += text.strip() + " "
                #this text is separated by speaker/pauses. For now, just appending all text to eachother. Later, split by speaker

        except Exception as e:
            print(f"Error during transcription: {e}")
            print("FileName:",audio_file)


        #add text to display
        time = datetime.datetime.now()
        if self.dialog_manager.find_by_time(time) is not None:
            # print("editing existing blurb!")
            self.dialog_manager.edit_by_time(time,text = transcribed_text)
        else:
            # print("adding new blurb!")
            self.dialog_manager.add_blurb(transcribed_text, start_time=time)

        if clearBuffer:
            self.dialog_manager.edit_by_time(time,text = transcribed_text,end_time= time)
            self.audio_buffer = np.array([], dtype=np.float32)
            self.silence_duration = 0.0  # Reset silence duration



    # Main function to handle incoming audio chunks
    def transcribe(self, sr, y):
        if(len(self.audio_buffer) == 0):    #more complex logic needed here to detect audio segmentation
            if(len(y) == 0):
                print("Your microphone is probably not working")
                return
            self.audio_buffer = y
            soundbyte_seconds = len(y)/sr
            self.current_buffer_start_time = datetime.datetime.now() - datetime.timedelta(seconds=soundbyte_seconds)

        else: 
            self.audio_buffer = np.concatenate([self.audio_buffer, y])

        buffer_duration = len(self.audio_buffer) / sr

        if buffer_duration >= self.max_buffer_duration:
            self.process_audio_buffer(self.model_large,sr,True)
        else:
            self.process_audio_buffer(self.model_small,sr,False)
        

    # Helper functions for Gradio interface
    def pre_prompt_words_display(self):
        return ', '.join(self.pre_prompt_words)

    def update_pre_prompt(self, words):
        new_words = [w.strip() for w in words.split(',') if w.strip()]
        self.pre_prompt_words = list(set(self.pre_prompt_words + new_words))
        self.save_pre_prompt_to_file()
        return self.pre_prompt_words_display()

    def remove_pre_prompt_word(self, word):
        self.pre_prompt_words = [w for w in self.pre_prompt_words if w != word]
        self.save_pre_prompt_to_file()
        return self.pre_prompt_words_display()

    def clear_text(self):
        self.dialog_manager.clear()




counter = 0
processor = TranscriptionProcessor()
executor = ThreadPoolExecutor(max_workers=1)

def transcribe_and_update(sr_y):
    global counter
    mine_counter = counter
    counter = counter + 1
    print("Begin",mine_counter)



    sr,y = sr_y
    processor.transcribe(sr,y)

    print("End counter",mine_counter)
    return processor.dialog_manager.to_string()

def ui():

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                audio = gr.Audio(streaming=True, label="Speak Now")

        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                output = gr.Textbox(
                    lines=25,
                    label="Transcription",
                    value="",
                    interactive=False,
                    autoscroll=True
                )
                with gr.Row():
                    clear_btn = gr.Button("Clear Text")
                    confirm_btn = gr.Button("Confirm delete", variant="stop", visible=False)
                    cancel_btn = gr.Button("Cancel", visible=False)

                    task_selection = gr.Radio(
                        label='Select Task',
                        choices=['transcribe', 'translate'],
                        value='transcribe'
                    )
                    # whisper_language_dropdown = gr.Dropdown(
                    #     label='Language',
                    #     value='english',
                    #     choices=['Auto-Detect'] + list(LANGUAGES.keys())
                    # )

        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                pre_prompt_input = gr.Textbox(
                    label="Add Pre-prompt Words (comma-separated)"
                )
                update_pre_prompt_button = gr.Button(value="Update Pre-prompt")

        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                pre_prompt_word_buttons = gr.Dataset(
                    components=[gr.Textbox(visible=False)],
                    samples=[[word] for word in processor.pre_prompt_words],
                    label="Click to remove pre-prompt word"
                )


        audio.change(
            fn=transcribe_and_update,
            inputs=audio,
            outputs=output,
            concurrency_limit=1,
            show_progress=False
        )

        # Clear text button handlers
        def show_confirm_buttons():
            return [gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)]

        def hide_confirm_buttons():
            return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)]

        def confirm_clear_text():
            processor.clear_text()
            return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), ""]

        clear_btn.click(
            fn=show_confirm_buttons,
            inputs=None,
            outputs=[clear_btn, confirm_btn, cancel_btn]
        )
        cancel_btn.click(
            fn=hide_confirm_buttons,
            inputs=None,
            outputs=[clear_btn, confirm_btn, cancel_btn]
        )
        confirm_btn.click(
            fn=confirm_clear_text,
            inputs=None,
            outputs=[clear_btn, confirm_btn, cancel_btn, output]
        )

        # Task and language selection handlers
        task_selection.change(
            fn=lambda task: processor.change_task(task),
            inputs=task_selection,
            outputs=None
        )
        # whisper_language_dropdown.change(
        #     fn=lambda lang: processor.change_whisper_language(lang),
        #     inputs=whisper_language_dropdown,
        #     outputs=None
        # )

        # Pre-prompt update handlers
        def update_pre_prompt(words):
            processor.update_pre_prompt(words)
            updated_samples = [[word] for word in processor.pre_prompt_words]
            return gr.update(samples=updated_samples)

        def remove_pre_prompt_word(word):
            processor.remove_pre_prompt_word(word[0])
            updated_samples = [[word] for word in processor.pre_prompt_words]
            return gr.update(samples=updated_samples)

        update_pre_prompt_button.click(
            fn=update_pre_prompt,
            inputs=pre_prompt_input,
            outputs=pre_prompt_word_buttons
        )
        pre_prompt_word_buttons.click(
            fn=remove_pre_prompt_word,
            inputs=pre_prompt_word_buttons,
            outputs=pre_prompt_word_buttons
        )

        # Update pre-prompt word buttons on load
        def update_gradio_elements():
            return gr.update(samples=[[word] for word in processor.pre_prompt_words])

        demo.load(
            fn=update_gradio_elements,
            inputs=None,
            outputs=pre_prompt_word_buttons
        )

    return demo

if __name__ == "__main__":
    demo = ui()
    demo.launch(server_port=7888, inbrowser=True)



#TODO

#verify asynchronous function calling


#swap to faster whisper

#figure out how to split audio chunks better.
# - using audio clustering
# - using silence detection
# - check if text output is different from previous transcription output
# - see what other people do

# - algorithms for sliding windows?


# # - use pyannote for speaker transition detection... and potentially speaker diarization
# from pyannote.audio import Pipeline
# pipeline = Pipeline.from_pretrained("pyannote/speaker-change-detection")
# audio_file = "./audio/file.wav"

# speaker_change = pipeline(audio_file)

# for speech_turn in speaker_change.get_timeline():
#     start_time = speech_turn.start
#     end_time = speech_turn.end
#     print(f"Speaker change detected from {start_time:.2f}s to {end_time:.2f}s")


#mel frequency cepstrum. <-- voice activity detector


#i think ONLY the large transcription chunk should be done asynchronously. 
#keep using small transcription on every piece of audio that comes through to keep up with the backlog

#... i really need a specialized output formatter for real for real on god.