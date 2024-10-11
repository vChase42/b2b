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
        #folder locations
        self.audio_folder = "audio"
        self.audio_segments_folder = "audio_segments"

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
        self.completing_last_row = False
        self.pre_prompt_words = []

        self.dialog_manager = DialogManager()

        #lock
        self.lock = Lock()

    # Load pre-prompt words from file

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
    def process_audio_buffer(self, sample_rate):
        buffer_start_time = self.current_buffer_start_time

        if len(self.audio_buffer) == 0:    #MAKE THIS BETTER
            return  # Nothing to process

        # Save audio to file
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")       
        audio_file = f"./{self.audio_folder}/{timestamp}.wav"
        os.makedirs(f'./{self.audio_folder}', exist_ok=True)
        sf.write(audio_file, self.format_audio_stream(self.audio_buffer), sample_rate)
        preprompt = f"{' '.join(self.pre_prompt_words)} {self.dialog_manager.get_text_before_time(datetime.datetime.now())[-500:]}"
        
        
        #split audio by speaker
        output_folder = f"./{self.audio_segments_folder}"
        os.makedirs(output_folder, exist_ok=True)
        speakers, filenames, times = diarize(self.pipeline, audio_file, output_folder)

        startendtime = None
        speaker = None
        audio_part = None

        if(len(times)) == 0:
            print("no speech found")
            return


        clear_buffer = len(self.audio_buffer) / sample_rate > self.max_buffer_duration
        
        #this try block handles some complexity.
        #if one speaker is detected, process with small model and update text display.
        #if MORE THAN ONE SPEAKER is detected, extract the very first speakers audio, process it with large.
        #delete first speakers audio from audio_buffer, leaving the rest of the audio intact. Update current_buffer_start_time
        #once an asynchronous large transcription task is submitted, function then uses small model to transcribe the NEXT speaker chunk.
        #end function.
        try:
            if(len(times) > 1) or clear_buffer:
                #do a large-model transcription, then clear buffer up to the start of the NEXT speech
                startendtime = times[0]
                speaker = speakers[0]
                audio_part = filenames[0]
                executor.submit(self.async_transcribe,self.model_large,audio_part,preprompt,buffer_start_time, startendtime, speaker)

                if clear_buffer:
                    self.audio_buffer = np.array([],dtype=np.float32)
                    return

                speech_start_timestamp_seconds, speech_end_timestamp_seconds = startendtime
                speech_end_index = int(speech_end_timestamp_seconds * sample_rate)
                self.current_buffer_start_time = buffer_start_time + datetime.timedelta(seconds=speech_end_timestamp_seconds)
                self.audio_buffer = self.audio_buffer[speech_end_index:]
                self.completing_last_row = True

                #small model transcription
                startendtime = times[1]
                speaker = speakers[1]
                audio_part = filenames[1]
                text = transcribe_audio(audio_part, self.model_small, preprompt).strip()

            elif(len(times) == 1):
                #small model transcription
                startendtime = times[0]
                speaker = speakers[0]
                audio_part = filenames[0]
                text = transcribe_audio(audio_part, self.model_small, preprompt).strip()

                if(startendtime == None):
                    print("critical error")
            else:
                print("something has gone wrong, no data to transcribe!")
                return
        except Exception as e:
            print("CRITICAL Error:",e)
            print("List size:",len(times))


        #add text to display
        start,end = startendtime
        start = buffer_start_time + datetime.timedelta(seconds=start)
        end = buffer_start_time + datetime.timedelta(seconds=end)
        middle = (end-start)/2 + start

        print("SMALL:",text)

        #if blurb exists
        if self.dialog_manager.find_by_time(middle) is not None and self.completing_last_row is False:
            self.dialog_manager.edit_by_time(middle,text = text, speaker_name=speaker)
        else:
            self.completing_last_row = False
            self.dialog_manager.add_blurb(text, start_time=start, speaker_name=speaker)




    def async_transcribe(self, model, audiofile, initial_prompt, buffer_start_time, startendtime, speaker = None):
        text = transcribe_audio(audiofile, model, initial_prompt).strip()
        start,end = startendtime
        start = buffer_start_time + datetime.timedelta(seconds=start)
        end = buffer_start_time + datetime.timedelta(seconds=end)
        middle = (end-start)/2 + start

        print("LARGE:",text)

        if self.dialog_manager.find_by_time(middle) is not None:
            self.dialog_manager.edit_by_time(middle,text = text,start_time=start,end_time=end, speaker_name = speaker)
        else:
            # print("adding blurb LARGE")
            self.dialog_manager.add_blurb(text,start_time=start,end_time=end, speaker_name=speaker)



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

        # buffer_duration = len(self.audio_buffer) / sr

        self.process_audio_buffer(sr)
        

    #pre prompt management functions
    def load_pre_prompt_from_file(self):
        if os.path.exists(self.pre_prompt_file):
            with open(self.pre_prompt_file, 'r') as file:
                self.pre_prompt_words = json.load(file)

    # Save pre-prompt words to file
    def save_pre_prompt_to_file(self):
        with open(self.pre_prompt_file, 'w') as file:
            json.dump(self.pre_prompt_words, file)

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
    print("Begin",mine_counter,"------------------------------------")

    sr,y = sr_y
    processor.transcribe(sr,y)

    # print("End counter",mine_counter)
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


#swap to faster whisper - check

#figure out how to split audio chunks better.
# - using audio clustering
# - using silence detection
# - check if text output is different from previous transcription output
# - see what other people do

# - algorithms for sliding windows?


# # - use pyannote for speaker transition detection... and potentially speaker diarization  - check
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

#... i really need a specialized output formatter for real for real on god. - check


#notes for resync
#mel frequency ceptstrum is a no-go, it basically returns an eigenmatrix
#pyannote has great diarization feature model. - but you need a key for it

#current computing issue: if i diarize and chunk the audio by silence/speaker, then i must do multiple whisper transcription computations.
#this is okay with small, but becomes noticeable with large (which triggeres every 10 seconds currently)
#the more speakers there are, the more the audio gets chunked, which means more .5 second transcription times (with large-v3)

#i removed concurrency, so i made it blocking again to make testing more easy, which caused the gaps to be more noticeable.
#solution: keep small-transcriptions blocking and sequential, and make ONLY large-v3 computations threaded.

#next feautures:
#reset audio buffer as soon as new speaker or silence detected

#diarization exists, but its not exactly like the model is keeping track of speakers between transcriptions
#chatgpt solution, extract embeddings from pyannote and then cluster them manually? seems complicated but doable. 


#implement better silence detection