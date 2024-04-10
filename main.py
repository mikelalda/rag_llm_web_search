from web_search import *
import gradio as gr
import torch
from rag_llm import *
from twitch_recorder import TwitchRecorder

DESCRIPTION = """\
# RAG LLM with web search
In this project I will try to add create a LLM chat with web search automatic dataset. After that try to create a chat with web search.
"""

LICENSE = """
<p/>
---
Depending on the model selection.
"""
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo may not work on CPU.</p>"

else:
    count = torch.cuda.device_count()
    names = dict()
    for i in range(count):
        names[str(i)]=torch.cuda.get_device_name(i)
    # torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    DESCRIPTION += "\n<p>Running on GPU 🥳</p>"

chat_llm = RAG_LLM()
chat_llm.start()

def load_twitch_data():
    return

def load_data(data_file):
    chat_llm.load_data(data_file)
    if chat_llm.status == 0:
        exit()

def respond_chat(history, msg):
    chat_llm.msg=msg
    chat_llm.respond()
    history.append((msg, chat_llm.response))
    if chat_llm.status.value == 0:
        exit()
    return history, ''

def interface():
    model = gr.Textbox(label="model", value="mistral", visible=False)
    with gr.Tab("Text2text"):
        with gr.Row("Input selection"):
            data_file = gr.File(type="filepath", label="Upload file for data analysis")   
        with gr.Row("Output ppt"):
            msg = gr.Textbox(label="Prompt") 

    msg.submit(fn=respond_chat, inputs=[chatbot, msg], outputs=[chatbot, msg])
    data_file.change(fn=load_data, inputs=[data_file])
        



with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    chatbot = gr.Chatbot()
    interface()
    gr.Markdown(LICENSE)

if __name__ == "__main__":
    demo.queue(max_size=20).launch()