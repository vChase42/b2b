from dotenv import load_dotenv
import os
load_dotenv()
hf_key = os.getenv('HF_KEY')

from model_utils import DiarizationManager, transcribe_audio, is_speech
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
import time



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
        #pyannote
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_key)
        self.pipeline.to(device)
        #faster whisper
        self.model_large = WhisperModel("large-v3", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
        self.model_small = WhisperModel("small", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
        #silero_vad
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        (self.get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

        #internal state
        self.diarization_manager = DiarizationManager(self.pipeline)
        self.dialog_manager = DialogManager()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_duration = 0.0
        self.current_buffer_start_time = None
        self.sample_rate = 0
        #debug info
        self.transcribe_time_running_average_large = 0
        self.transcribe_time_running_average_small = 0
        self.transcribe_time_running_average_diarize = 0

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
        preprompt = f"{' '.join(self.pre_prompt_words)} {self.dialog_manager.get_text_before_time(self.current_buffer_start_time)[-250:]}"

        if len(self.audio_buffer) == 0:
            print("No audio detected.")
            return

        audio_file = self.save_wav_file(self.audio_buffer)

        #use VAD
        if not is_speech(self.vad_model, self.get_speech_timestamps, audio_file):
            print("silero VAD did not detect any speech")
            self.audio_buffer = []
            self.current_buffer_start_time = None
            return

        #diarize
        start_time = time.time()
        diarized_dicts = self.diarization_manager.diarize(audio_file, self.audio_segments_folder)
        elapsed_time = time.time() - start_time
        self.transcribe_time_running_average_diarize = self.running_average(self.transcribe_time_running_average_diarize, elapsed_time)

        if len(diarized_dicts) == 0:
            print("Diarization attempt failed, no speakers detected")
            self.audio_buffer = []
            self.current_buffer_start_time = None
            return

        #transcribe
        #if there is 1 seconds of silence after end_seconds, then true
        done_speaking_flag = len(diarized_dicts) == 1 and 1 < self.buffer_duration - diarized_dicts[0]['end_seconds']  
        print("NUMBER OF AUDIO SEGMENTS:",len(diarized_dicts))
        #metric for done-ness can be number of words transcribed so far. more than 5 words = good

        if len(diarized_dicts) > 1 or done_speaking_flag:
            # print("popping! program thinks speaker is done speaking:",done_speaking_flag)
            # print(f"buffer duration is {self.buffer_duration}, and seconds timestamp last spoken is {diarized_dicts[0]['end_seconds']}")
            diarized_info = diarized_dicts.pop(0)

            #buffer management
            speech_end_index = int(diarized_info['end_seconds'] * self.sample_rate)
            self.audio_buffer = self.audio_buffer[speech_end_index:]
            start_time = buffer_start_time + datetime.timedelta(seconds=diarized_info['start_seconds'])
            end_time = buffer_start_time + datetime.timedelta(seconds=diarized_info['end_seconds'])
            self.current_buffer_start_time = end_time

            #transcribe row LARGE
            self.dialog_manager.finalize_latest_row(start_time, end_time)  #what happens when we try to finalize a row that hasnt been even made yet
            self.executor.submit(self.transcribe_update_text, diarized_info, buffer_start_time, preprompt, self.model_large)
            
            if(len(diarized_dicts) == 0): return

        #transcribe rest of buffer
        self.executor.submit(self.transcribe_update_text(diarized_dicts[0], buffer_start_time, preprompt, self.model_small))


    def save_wav_file(self, y):
        # Save audio to file
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")       
        audio_file = f"./{self.audio_folder}/{timestamp}.wav"
        # max_val = np.max(np.abs(y))
        # if max_val != 0:
        #     y = y.astype(np.float32)
        #     y /= max_val
        sf.write(audio_file, y, self.sample_rate)        
        
        return audio_file
    
    def transcribe_update_text(self, diarize_dict, buffer_start_time, preprompt, model):
        DiarizationManager.print_dict(diarize_dict)
        start_time = time.time()
        text = transcribe_audio(diarize_dict['audiofile'], model, preprompt)
        elapsed_time = time.time() - start_time
        
        #record running average, debug info.
        if(model == self.model_small):
            self.transcribe_time_running_average_small = self.running_average(self.transcribe_time_running_average_small, elapsed_time)
        else:
            self.transcribe_time_running_average_large = self.running_average(self.transcribe_time_running_average_large, elapsed_time)

        text = text + f" ({elapsed_time})"
        self.update_text(diarize_dict,buffer_start_time,text, speaker = diarize_dict['speaker'])

    def update_text(self,diarize_dict, buffer_start_time, text, speaker = None):
        start = diarize_dict['start_seconds']
        end = diarize_dict['end_seconds']
        start = buffer_start_time + datetime.timedelta(seconds=start)
        end = buffer_start_time + datetime.timedelta(seconds=end)
        middle = (end-start)/2 + start

        #if blurb exists
        if self.dialog_manager.find_by_time(middle) is not None:
            self.dialog_manager.edit_by_time(middle,text = text, speaker_name=speaker)
        else:
            self.dialog_manager.add_blurb(text, start_time=start, speaker_name=speaker)

    def get_text(self):
        return self.dialog_manager.to_string()



    def running_average(self,current_running_average, new_transcribe_time, alpha=0.125):
        new_average = (1 - alpha) * current_running_average + alpha * new_transcribe_time
        return new_average



    #pre prompt management functions
    def load_pre_prompt_from_file(self):
        if os.path.exists(self.pre_prompt_file):
            with open(self.pre_prompt_file, 'r') as file:
                self.pre_prompt_words = json.load(file)
        else:
            self.pre_prompt_words = []

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


