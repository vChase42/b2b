from dotenv import load_dotenv
import os
load_dotenv()
hf_key = os.getenv('HF_KEY')

from split_and_transcribe import diarize, transcribe_audio
from dialog_manager import DialogManager
from pyannote.audio import Pipeline

import soundfile as sf
from faster_whisper import WhisperModel
import torch
import numpy as np
import json
import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Lock



class TranscriptionProcessor:

    def __init__(self):
        #folder locations
        self.audio_folder = "audio"
        self.audio_segments_folder = "audio_segments"
        os.makedirs(self.audio_folder, exist_ok=True)
        os.makedirs(self.audio_segments_folder, exist_ok=True)


        #preprompt loc
        self.pre_prompt_file = 'pre-prompt.json'
        self.load_pre_prompt_from_file()


        if not torch.cuda.is_available():
            print("-------------------------------")
            print("WARNING: CUDA is not available!")
            print("-------------------------------")

        #models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_key)
        self.pipeline.to(device)
        self.model_large = WhisperModel("large-v3", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
        self.model_small = WhisperModel("small", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")

        #internal state
        self.dialog_manager = DialogManager()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_duration = 0.0
        self.current_buffer_start_time = None
        self.sample_rate = 0

        #lock
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.lock = Lock()

    def update_audio_buffer(self, sr,y):
        if(len(y) == 0):
            print("Your microphone is probably not working")
            return
        
        if(len(self.audio_buffer) == 0):
            self.audio_buffer = y
        else: 
            self.audio_buffer = np.concatenate([self.audio_buffer, y])

        self.buffer_duration = len(self.audio_buffer)/sr
        self.current_buffer_start_time = datetime.datetime.now() - datetime.timedelta(seconds=self.buffer_duration)
        self.sample_rate = sr

    def main_processing_pipeline(self):
        buffer_start_time = self.current_buffer_start_time
        preprompt = f"{' '.join(self.pre_prompt_words)} {self.dialog_manager.get_text_before_time(datetime.datetime.now())[-250:]}"

        # if not speech_detected(audio_buffer):
        #     audio_buffer = None
        #     buffer_start = now
        #     return

        if len(self.audio_buffer) == 0:
            print("No audio detected.")
            return

        audio_file = self.save_wav_file(self.audio_buffer)
        
        diarized_dicts = diarize(self.pipeline, audio_file, self.audio_segments_folder, limit = 3000)
        if len(diarized_dicts) == 0:
            print("Diarization attempt failed, no speakers detected")
            return


        
        done_speaking_flag = len(diarized_dicts) == 1 and 2 < self.buffer_duration - diarized_dicts[0]['end_seconds']  #if there is 2 seconds of silence after end_seconds, then true
        if len(diarized_dicts) > 1 or done_speaking_flag:
            # print("popping! program thinks speaker is done speaking:",done_speaking_flag)
            # print(f"buffer duration is {self.buffer_duration}, and seconds timestamp last spoken is {diarized_dicts[0]['end_seconds']}")
            diarized_info = diarized_dicts.pop(0)

            #buffer management
            speech_end_index = int(diarized_info['end_seconds'] * self.sample_rate)
            self.audio_buffer = self.audio_buffer[speech_end_index:]
            end_time = buffer_start_time + datetime.timedelta(seconds=diarized_info['end_seconds'])
            self.current_buffer_start_time = end_time

            #transcribe row LARGE
            self.dialog_manager.finalize_latest_row(end_time)
            self.executor.submit(self.transcribe_finalize, diarized_info, buffer_start_time, preprompt)
            


            if(len(diarized_dicts) == 0): return


        text = transcribe_audio(diarized_dicts[0]['audiofile'], self.model_small, preprompt)
        self.update_text(diarized_dicts[0],buffer_start_time,text)


    def save_wav_file(self, y):
        # Save audio to file
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")       
        audio_file = f"./{self.audio_folder}/{timestamp}.wav"
        max_val = np.max(np.abs(y))
        if max_val != 0:
            y = y.astype(np.float32)
            y /= max_val
        sf.write(audio_file, y, self.sample_rate)        
        
        return audio_file
    
    def transcribe_finalize(self, diarize_dict, buffer_start_time, preprompt):
        text = transcribe_audio(diarize_dict['audiofile'], self.model_large, preprompt)
        self.update_text(diarize_dict,buffer_start_time,text)



    def update_text(self,diarize_dict, buffer_start_time, text):
        
        start = diarize_dict['start_seconds']
        end = diarize_dict['end_seconds']
        start = buffer_start_time + datetime.timedelta(seconds=start)
        end = buffer_start_time + datetime.timedelta(seconds=end)
        middle = (end-start)/2 + start

        #if blurb exists
        if self.dialog_manager.find_by_time(middle) is not None:
            self.dialog_manager.edit_by_time(middle,text = text)
        else:
            self.dialog_manager.add_blurb(text, start_time=start)

    def get_text(self):
        return self.dialog_manager.to_string()



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

