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
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

else:
    torch.cuda.set_device(1)
    torch.cuda.empty_cache()
    DESCRIPTION += "\n<p>Running on GPU 🥳</p>"

chat_llm = RAG_LLM()


def respond_chat(history, msg):
    chat_llm.respond_LLM(msg)
    history += chat_llm.response
    return history, history, ''
def interface():
    history = gr.Textbox(label="history", visible=False)
    tokenizer = gr.Textbox(label="tokenizer", value="google/gemma-7b", visible=False)
    model = gr.Textbox(label="model", value="google/gemma-7b", visible=False)
    with gr.Tab("Text2text"):
        with gr.Row("Input selection"):
            data_file = gr.File(type="filepath", label="Upload file for data analysis")   
        with gr.Row("Output ppt"):
            msg = gr.Textbox(label="Prompt") 

    msg.submit(fn=respond_chat, inputs=[history, msg], outputs=[history, chatbot, msg])
    data_file.change(fn=chat_llm.load_data, inputs=[data_file,tokenizer,model])
        



with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(value="Duplicate Space for private use", elem_id="duplicate-button")
    chatbot = gr.Chatbot()
    interface()
    gr.Markdown(LICENSE)

if __name__ == "__main__":
    demo.queue(max_size=20).launch()