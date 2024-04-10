import torch
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from transformers import BitsAndBytesConfig
from threading import Thread
import PyPDF2
import os
import logging
import datetime
import enum

class LLMStatus(enum.Enum):
    ERROR = 0
    INITIALIZING = 1
    LOADING = 2
    RUNNING = 3
    STOPPED = 4

class RAG_LLM(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.status = LLMStatus.INITIALIZING
        self.qa = None
        self.docs = []
        modelPath = "sentence-transformers/all-MiniLM-l6-v2"
        model_kwargs = {'device':'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=modelPath, 
            model_kwargs=model_kwargs, 
            encode_kwargs=encode_kwargs 
            )
        self.model = ChatOllama(model="mistral")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
             maximum and keep the answer concise. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )
        self.msg = ''
        self.response = ''
        
        logging.basicConfig(filename="logs/rag_llm.log", level=logging.INFO)
        # logging.getLogger().addHandler(logging.StreamHandler())
        logging.info("Starting de app at %s_%s", datetime.date.fromisoformat('2019-12-04'), datetime.datetime.now().strftime("%H:%M:%S"))
        self.status = LLMStatus.RUNNING
        
    def load_data(self, dataset):
        try:
            logging.info('Loading data, please wait...')
            self.status = LLMStatus.LOADING
            self.dataset = dataset
            if dataset != None:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                if self.dataset.split('.')[-1] == 'csv':
                    loader = CSVLoader(file_path=self.dataset, encoding="utf-8", csv_args={'delimiter': ','})
                    data = loader.load()
                    self.docs.extend(text_splitter.split_documents(data))
                    self.db = FAISS.from_documents(self.docs, self.embeddings)
                elif self.dataset.split('.')[-1] == 'pdf':
                    data = ""
                    with open(self.dataset, 'rb') as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        num_pages = len(pdf_reader.pages)

                        # Convert the content of each page to a string.
                        text = ""
                        for page_num in range(num_pages):
                            page = pdf_reader.pages[page_num]
                            text += page.extract_text()
                        data += text
                        data += "\n"
                    self.docs.extend(text_splitter.split_text(data))
                    self.db = FAISS.from_texts(self.docs, self.embeddings)
                logging.info('Data loaded')
                self.status = LLMStatus.RUNNING
        except Exception as e:
            logging.error(e)
            self.status = LLMStatus.ERROR
            logging.info("Unexpected error happened at %s. App will close", datetime.datetime.now().strftime("%Hh%Mm%Ss"))

    def respond(self):
        try:
            if self.status == LLMStatus.RUNNING:
                if self.qa == None:
                    pipe = Ollama(model="llama2")
                    self.response = pipe.invoke(self.msg)
                else:
                    self.chain = ({"context": self.db, "question": RunnablePassthrough()}
                      | self.prompt 
                      | self.model
                      | StrOutputParser())
                    self.response = self.chain.invoke({"query": self.msg})
            else:
                logging.info('Please wait until model is loaded, select at least 1 document as retrieval data')
        except Exception as e:
            logging.error(e)
            self.status = LLMStatus.ERROR
            logging.info("Unexpected error happened at %s. App will close", datetime.datetime.now().strftime("%Hh%Mm%Ss"))


class RAG_LLM_B(Thread):
    def __init__(self):
        self.status = LLMStatus.INITIALIZING
        Thread.__init__(self)
        self.qa = None
        self.docs = []
        modelPath = "sentence-transformers/all-MiniLM-l6-v2"
        model_kwargs = {'device':'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=modelPath, 
            model_kwargs=model_kwargs, 
            encode_kwargs=encode_kwargs 
            )
        self.msg = ''
        self.response = ''
        
        logging.basicConfig(filename="logs/rag_llm.log", level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info("Starting de app at %s", datetime.datetime.now().strftime("%Hh%Mm%Ss"))
        
    def load_data(self, dataset, tokenizer, model):
        try:
            logging.info('Loading data, please wait...')
            self.status = LLMStatus.LOADING
            self.dataset = dataset
            if dataset != None:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                if self.dataset.split('.')[-1] == 'csv':
                    loader = CSVLoader(file_path=self.dataset, encoding="utf-8", csv_args={'delimiter': ','})
                    data = loader.load()
                    self.docs.extend(text_splitter.split_documents(data))
                    self.db = FAISS.from_documents(self.docs, self.embeddings)
                elif self.dataset.split('.')[-1] == 'pdf':
                    data = ""
                    with open(self.dataset, 'rb') as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        num_pages = len(pdf_reader.pages)

                        # Convert the content of each page to a string.
                        text = ""
                        for page_num in range(num_pages):
                            page = pdf_reader.pages[page_num]
                            text += page.extract_text()
                        data += text
                        data += "\n"
                    self.docs.extend(text_splitter.split_text(data))
                    self.db = FAISS.from_texts(self.docs, self.embeddings)
                logging.info('Loading the model, please wait...')
                self.load_qa_model( tokenizer, model, True)
        except Exception as e:
            logging.error(e)
            self.status = LLMStatus.ERROR
            logging.info("Unexpected error happened at %s. App will close", datetime.datetime.now().strftime("%Hh%Mm%Ss"))
            
    def load_qa_model(self, tokenizer, model, dataset=False):
        try:
            token = os.getenv("HF_TOKEN")
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, load_in_8bit_fp32_cpu_offload=True)
            model_4bit = AutoModelForCausalLM.from_pretrained(model, device_map='auto', quantization_config=quantization_config)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, device_map='auto', padding=True, truncation=True, max_length=512)
            self.model = AutoModelForCausalLM.from_pretrained(model_4bit)
            pipe = pipeline(
                "text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer,
                return_tensors='pt',
                max_length=512,
                max_new_tokens=512,
                model_kwargs={"torch_dtype": torch.bfloat16},
                # device="cuda"
                )

            llm = HuggingFacePipeline(
                pipeline=pipe,
                model_kwargs={"temperature": 0.7, "max_length": 512},
                )
            if dataset:
                self.qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=self.db.as_retriever()
                    )
                self.status = LLMStatus.RUNNING
                logging.info('Model loaded')
            else:
                logging.info('Please select at least 1 document as retrieval data')

        except Exception as e:
            logging.error(e)
            self.status = LLMStatus.ERROR
            logging.info("Unexpected error happened at %s. App will close", datetime.datetime.now().strftime("%Hh%Mm%Ss"))

    def respond(self):
        try:
            if self.status == LLMStatus.RUNNING:
                if self.qa == None:
                    logging.info('Please select at least 1 document as retrieval data')
                    self.response = 'Please select at least 1 document as retrieval data'
                else:
                    self.response = self.qa({"query": self.msg})
            else:
                logging.info('Please wait until model is loaded, select at least 1 document as retrieval data')
        except Exception as e:
            logging.error(e)
            self.status = LLMStatus.ERROR
            logging.info("Unexpected error happened at %s. App will close", datetime.datetime.now().strftime("%Hh%Mm%Ss"))
