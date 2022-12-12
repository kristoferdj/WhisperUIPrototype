import gradio as gr
import whisper
import langs

langs = sorted(langs.LANGUAGES.values())


def transcribe(text, device, task, language, model_size, file, mic):
    if device == 'gpu':
        device = 'cuda'
    args = {'task': task}
    if (model_size == 'tiny.en') or (model_size == 'base.en') or (model_size == 'small.en') or (
            model_size == 'medium.en'):
        args['language'] = 'english'
    elif (language == 'Detect'):
        args['language'] = None
    else:
        args['language'] = language
    model = whisper.load_model(model_size, device)
    if mic is not None:
        audio = mic
    elif file is not None:
        audio = file
    else:
        return "You must provide a mic recording or file"
    result = model.transcribe(audio, **args)
    return result["text"]


demo = gr.Interface(transcribe,
                    inputs=[
                        gr.Textbox(label="", value="Capstone Design Group 4 prototype. Features subject to change."),
                        gr.Radio(['gpu', 'cpu'], label='Processor'),
                        gr.Radio(['transcribe', 'translate'], label='Task'),
                        gr.Dropdown(langs, value='Detect', label='Language'),
                        gr.Dropdown(['tiny', 'base', 'small', 'medium', 'large'], value='small', label='Model'),
                        gr.Audio(source='upload', type='filepath', optional=True, label='Audio'),
                        gr.Audio(label='Microphone', source='microphone', type='filepath'),
                    ],
                    css='*{color: green}',
                    outputs="text"

                    )
demo.launch(share=True)
