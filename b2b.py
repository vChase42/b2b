import os
import gradio as gr
from transcription_processor import TranscriptionProcessor


processor = TranscriptionProcessor()
counter = 0

def transcribe_and_update(sr_y):
    if(sr_y is None): return
    global counter, processor
    print("Begin",counter,"------------------------------------")
    counter = counter + 1

    sr,y = sr_y
    processor.update_audio_buffer(sr,y)
    processor.main_processing_pipeline()

    # print("End counter",mine_counter)
    return processor.get_text()

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

    print("launching UI")
    demo = ui()
    demo.launch(server_port=7888, inbrowser=True)



#TODO

#need to deal better with the fuzze edge case of silences. to put the problem in words...
#when there is noone speaking, the buffer queue doesnt get processed. 
#only when people are talking, does the queue actually get processed. Otherwise, its frozen in a static state.

#what should happen, is if there is noone speaking, the timer should continue ticking, and given enough silence, 
#the whole buffer should just be transcribed and cleared



#diarization exists, but its not exactly like the model is keeping track of speakers between transcriptions
#chatgpt solution, extract embeddings from pyannote and then cluster them manually? seems complicated but doable. 
#implement better silence detection


