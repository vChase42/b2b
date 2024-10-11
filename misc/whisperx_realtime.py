import os
import json
import time
import datetime
import threading
import webrtcvad
import numpy as np
import gradio as gr
import noisereduce as nr
import soundfile as sf
from collections import deque
from pydub import AudioSegment
import whisperx
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

LANGUAGES = whisperx.utils.TO_LANGUAGE_CODE
whisper_language = 'english'

# Create a ThreadPoolExecutor with 16 threads to transcribe the audio in parallel
executor = ThreadPoolExecutor(max_workers=16)

# Create a queue for transcription tasks
transcription_queue = Queue()

running_avg_buffer = deque(maxlen=30)
average_threshold_ratio = 0.5

vad = webrtcvad.Vad()
vad.set_mode(3)

audio_stream = np.array([])
text_stream = [""]

# text_lock = threading.Lock()

min_duration = 3
max_duration = 10

# counter for skipping frames in the audio stream 
counter = 0
counter_skips = 2

# Index for the transcriptions in the text stream array
transcript_index = 0

pre_prompt_words = []
pre_prompt_file = 'pre-prompt.json'

# task to be performed by the model (transcribe/translate)
current_task = 'transcribe'


def save_pre_prompt_to_file():
    global pre_prompt_words,  pre_prompt_file
    with open(pre_prompt_file, 'w') as file:
        json.dump(pre_prompt_words, file)

def load_pre_prompt_from_file():
    global pre_prompt_words,  pre_prompt_file
    if os.path.exists(pre_prompt_file):
        with open(pre_prompt_file, 'r') as file:
            pre_prompt_words = json.load(file)

# Load pre-prompt words when the script starts
load_pre_prompt_from_file()

# Create a transcribers
transcriber_large = whisperx.load_model('large-v3', device='cuda', asr_options = {'initial_prompt': ', '.join(pre_prompt_words)})
transcriber_small=  transcriber_large #whisperx.load_model("base.en", device='cuda', asr_options = {'initial_prompt': ', '.join(pre_prompt_words)})


def calculate_n_fft(time_duration_ms, sample_rate):
    time_duration_s = time_duration_ms / 1000.0  # Convert milliseconds to seconds
    n_fft = time_duration_s * sample_rate
    n_fft_power_of_two = 2**np.round(np.log2(n_fft))  # Round to nearest power of two
    return int(n_fft_power_of_two)

def reduce_noise_on_file(y, sr):
    # perform noise reduction
    time_duration_ms = 12  # milliseconds   
    n_fft = calculate_n_fft(time_duration_ms, sr)
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=y, sr=sr, stationary=False, thresh_n_mult_nonstationary=1.5,
                                    time_constant_s=1.0,  sigmoid_slope_nonstationary=2.0,  n_fft=n_fft, device='cuda')
    return reduced_noise
   

def vad_filter(y, sr):

    # Reduce noise in the audio data to improve VAD performance
    ry = reduce_noise_on_file(y, sr)

     # Convert the audio data to the correct format
    audio = AudioSegment(ry.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    audio = audio.set_frame_rate(16000)

    # save audio to file
    audio.export('vad.wav', format='wav')

    # Split the audio into 10 ms frames
    frame_duration_ms = 10  # Duration of a frame in ms
    bytes_per_sample = 2
    frame_byte_count = int(sr * frame_duration_ms / 1000) * bytes_per_sample  # Number of bytes in a frame
    frames = [audio.raw_data[i:i+frame_byte_count] for i in range(0, len(audio.raw_data), frame_byte_count)]

    voice_activity = False
    # Use VAD to check if each frame contains speech
    for frame in frames:
        if len(frame) != frame_byte_count:
            continue  # Skip frames that are not exactly 10 ms long
        
        contains_speech = vad.is_speech(frame, sample_rate=16000)
        
        if contains_speech:
            voice_activity = True
            break

    return voice_activity

def update_pre_prompt(words):
    global pre_prompt_words
    new_words = [w.strip() for w in words.split(',') if w.strip()]
    pre_prompt_words = list(set(pre_prompt_words + new_words))  # Remove duplicates
    save_pre_prompt_to_file()
    return  gr.update(samples=[[word] for word in pre_prompt_words])

def update_gradio_elements():
    return gr.update(samples=[[word] for word in pre_prompt_words])

def remove_pre_prompt_word(word):
    global pre_prompt_words
    word = word[0]  # Extract the string from the list
    pre_prompt_words = [w for w in pre_prompt_words if w != word]
    save_pre_prompt_to_file()
    return gr.update(samples=[[word] for word in pre_prompt_words])

def call_large_transcriber(sr):
    global audio_stream, min_duration
    size_samples = min_duration * sr
    if len(audio_stream) > size_samples:
        transcription_queue.put(('large', sr))
            

def transcribe_large_chunk(timestamp, stream, index, sr):
    global text_stream, audio_stream, pre_prompt_words, LANGUAGES, whisper_language, current_task

    tik = time.time()
    os.makedirs('./audio', exist_ok=True)
    audio_file = f"./audio/{timestamp}-large.wav"
    sf.write(audio_file, format_audio_stream(stream), sr)
    
    current_pre_prompt = " ".join(pre_prompt_words)
    new_options = transcriber_large.options._replace(initial_prompt=current_pre_prompt)
    transcriber_large.options = new_options
    
    language = LANGUAGES[whisper_language] if whisper_language in LANGUAGES else None
    segments = transcriber_large.transcribe(audio_file,language=language,task=current_task)['segments']
    results = [segment['text'] for segment in segments]

    transcribed_text_with_timestamp = f"[{timestamp}] {' '.join(results)}\n"
    text_stream[index] = transcribed_text_with_timestamp

    with open("transcriptions.txt", "a") as file:
        file.write(transcribed_text_with_timestamp)
        
    return time.time()-tik


def transcribe_small_chunk(timestamp, index, sr):
    global text_stream, pre_prompt_words, audio_stream, LANGUAGES, whisper_language, current_task
   
    tik = time.time()
    os.makedirs('./audio', exist_ok=True)
    audio_file = f'./audio/{timestamp}-small.wav'
    sf.write(audio_file, format_audio_stream(audio_stream), sr)
    
    # update the pre-prompt words 
    current_pre_prompt = " ".join(pre_prompt_words)
    new_options = transcriber_small.options._replace(initial_prompt=current_pre_prompt)
    transcriber_small.options = new_options
    
    language = LANGUAGES[whisper_language] if whisper_language in LANGUAGES else None
    segments = transcriber_small.transcribe(audio_file,language=language, task=current_task)['segments']
    results = [segment['text'] for segment in segments]
    
    result = ' '.join(results)
    text_stream[index] = f'\n<{result}>'
    
    return time.time()-tik


 # Add a new function to process the transcription queue
def process_transcription_queue():
    global  text_stream, audio_stream, transcript_index, counter, counter_skips
    while True:
        try:
            task = transcription_queue.get()
            if task is None:
                break
            task_type, *args = task
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        
        
            if task_type == 'large':
                stream = np.array(audio_stream)
                audio_stream = np.array([])
                text_stream.append("")
                    
                cost = transcribe_large_chunk(timestamp, stream, transcript_index, *args)
                print(f'{timestamp} - {task_type} -index {transcript_index}- {cost:.3f} seconds')
                transcript_index += 1
            
            elif task_type == 'small':
                counter += 1
                if (counter % counter_skips == 0): 
                    cost = transcribe_small_chunk(timestamp, transcript_index, *args)
                    print(f'{timestamp} - {task_type} -index {transcript_index}- {cost:.3f} seconds')
            transcription_queue.task_done()
            
        except Exception as e:
            print("Error in transcription queue", e)
# Start the queue processing thread
queue_thread = threading.Thread(target=process_transcription_queue)
queue_thread.start()
 

def update_running_average(new_value):
    running_avg_buffer.append(new_value)
    return np.mean(running_avg_buffer)


def is_silence(audio_data, base_threshold=150):
    """Check if the given audio data represents silence based on running average."""
    current_level = np.abs(audio_data).mean()
    running_avg = update_running_average(current_level)
    
    dynamic_threshold = running_avg * average_threshold_ratio
    threshold = max(dynamic_threshold, base_threshold)

    # print(f"Current Level: {current_level}, Running Avg: {running_avg}, Threshold: {threshold}")

    return current_level < threshold

def format_audio_stream(y):
    # Convert the audio to the correct format
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    return y

def reverse_format_audio_stream(y):
    # Find the maximum absolute value
    max_abs_value = np.max(np.abs(y))
    
    # Multiply by the max absolute value to undo the normalization
    y *= max_abs_value
    
    # Convert back to int16
    y = (y * 32767).astype(np.int16)
    
    return y

def transcribe(sr, y):
    global audio_stream, max_duration
    
    audio_stream = np.concatenate([audio_stream, y])
    stream = np.array(audio_stream)
    
    # if the stream is silent reset steam
    if np.sum(np.abs(stream)) == 0 or is_silence(stream):
        audio_stream = np.array([])
        return 
    
    # if only the last chunk is silent caLL large transcriber
    if np.sum(np.abs(y)) == 0 or is_silence(y):
        call_large_transcriber(sr)
        return
    
    if not vad_filter(y, sr):
        call_large_transcriber(sr)
        return

    if (len(stream) / sr) > max_duration:
        call_large_transcriber(sr)
    elif (len(stream) / sr) <= (max_duration * 0.80):
        transcription_queue.put(('small', sr))
        

def transcribe_and_update(new_chunk):
    global text_stream

    sr, y = new_chunk
    executor.submit(transcribe, sr, y)
    text = ' '.join(text_stream)
    return text

def clear_text():
    global text_stream, transcript_index
    text_stream = [""]
    transcript_index = 0
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), ""

def change_task(task):
    global current_task
    current_task = task
    
def change_whisper_language(language):
    global whisper_language
    whisper_language = language
    
def ui():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                audio = gr.Audio(streaming=True, label="Speak Now")

        with gr.Row():
            with gr.Column(scale=1, min_width=600 ):
                output = gr.Textbox(lines=25,  label="Transcription", autoscroll=True)
                with gr.Row():
                    delete_btn = gr.Button("Clear Text")
                    confirm_btn = gr.Button("Confirm delete", variant="stop", visible=False)
                    cancel_btn = gr.Button("Cancel", visible=False)
                    
                    task_selection = gr.Radio(label='Select Task', choices=['transcribe', 'translate'], value='transcribe')
                    whipser_language = gr.Dropdown(label='Language', value='english', choices=['Auto-Detect'] + list(LANGUAGES.keys()) )

        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                pre_prompt_input = gr.Textbox(label="Add Pre-prompt Words (comma-separated)")
                update_pre_prompt_button = gr.Button(value="Update Pre-prompt")

        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                pre_prompt_word_buttons = gr.Dataset(
                    components=[gr.Textbox(visible=False)],
                    samples=[],
                    label="Click to remove pre-prompt word"
                )

        audio.change(
            transcribe_and_update, 
            inputs=audio, 
            outputs=output, 
            concurrency_limit=1, 
            show_progress=False)
        
        # clear_button.click(clear_text, outputs=output)
        task_selection.change(change_task, task_selection, [])
        whipser_language.change(change_whisper_language, whipser_language, [])
        
        delete_btn.click(lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [delete_btn, confirm_btn, cancel_btn])
        cancel_btn.click(lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [delete_btn, confirm_btn, cancel_btn])
        confirm_btn.click(clear_text, None, [delete_btn, confirm_btn, cancel_btn, output])
        
        update_pre_prompt_button.click(
            update_pre_prompt,
            inputs=[pre_prompt_input],
            outputs=[ pre_prompt_word_buttons]
        )
        pre_prompt_word_buttons.click(
            remove_pre_prompt_word,
            inputs=[pre_prompt_word_buttons],
            outputs=[pre_prompt_word_buttons]
        )
        
        demo.load(update_gradio_elements, [], [pre_prompt_word_buttons])

    return demo

if __name__ == "__main__":
    demo = ui()
    demo.launch(server_port=7888, inbrowser=True)
