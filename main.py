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
chat_llm = RAG_LLM('llama2')

def load_model(model):
    chat_llm.model_change(model)

def load_twitch_data():
    return

def web_search(text, max_results):
    websearch = WebSearcher(text, "data/web_search_output.csv", max_results)
    websearch.start()
    websearch.search()
    websearch.join()
    chat_llm.load_data("data/web_search_output.csv")
    if chat_llm.status == 0:
        exit()

def load_data(data_file):
    if chat_llm != None:
        chat_llm.load_data(data_file)
        if chat_llm.status == 0:
            exit()

def respond_chat(history, msg, websearch, maxresults):
    
    chat_llm.msg=msg
    # if websearch:
    #     web_search(msg, maxresults)
    chat_llm.respond()
    history.append((msg, chat_llm.response))
    if chat_llm.status.value == 0:
        exit()
    return history, ''

def clear_inputs():
    chat_llm.db = None
    chat_llm.dataset = None
    chat_llm.docs = None
    msg = ''
    chatbot = []
    return chatbot, msg

def interface():
    model = gr.Dropdown(choices=["llama2", "mistral"], value="llama2", label="Model selection", info="Choose a model, as defaul llama2 selected")
    with gr.Tab("Text2text"):
        with gr.Row("Page cuantity selection"):
            websearch = gr.Checkbox(label="Web search")
            maxresults = gr.Slider(minimum=1, maximum=15, step=1, value=1, label="Number of web searches per text input")
        with gr.Row("Input selection"):
            data_file = gr.File(type="filepath", label="Upload file for data analysis")   
        with gr.Row("Output ppt"):
            msg = gr.Textbox(label="Prompt") 
        clear = gr.Button(value='clear all')
    
    msg.submit(fn=respond_chat, inputs=[chatbot, msg, websearch, maxresults], outputs=[chatbot, msg])
    data_file.change(fn=load_data, inputs=[data_file])
    model.change(fn=load_model,inputs=[model])
    clear.click(fn=clear_inputs,outputs=[chatbot, msg])
        



with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    chatbot = gr.Chatbot()
    interface()
    gr.Markdown(LICENSE)

if __name__ == "__main__":
    demo.queue(max_size=20).launch()