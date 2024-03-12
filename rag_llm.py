import torch
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import BitsAndBytesConfig

import PyPDF2
import os

os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING']='1'
class RAG_LLM:
    def __init__(self):
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

    def load_data(self, dataset, tokenizer, model):
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
            self.load_qa_model( tokenizer, model, True)
        

    
    def load_qa_model(self, tokenizer, model, dataset=False):
        token = os.getenv("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, device_map='auto', padding=True, truncation=True, max_length=512, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model, device_map='auto', token=token)
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

        self.llm = HuggingFacePipeline(
            pipeline=pipe,
            model_kwargs={"temperature": 0.7, "max_length": 512},
            )
        if dataset:
            self.qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.db.as_retriever()
                )
        
    
    def respond_LLM(self, msg):
        if self.qa == None:
            self.response = 'Please select at least 1 document as retrieval data'
        else:
            self.response = self.qa({"query": msg})

class RAG_LLM_B:
    def __init__(self):
        modelPath = "sentence-transformers/all-MiniLM-l6-v2"
        model_kwargs = {}
        encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=modelPath, 
            model_kwargs=model_kwargs, 
            encode_kwargs=encode_kwargs 
            )

    def load_data(self, dataset):
        self.dataset = dataset
        if dataset != None:

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            if self.dataset.split('.')[-1] == 'csv':
                loader = CSVLoader(file_path=self.dataset, encoding="utf-8", csv_args={'delimiter': ','})
                data = loader.load()
                docs = text_splitter.split_documents(data)
                self.db = FAISS.from_documents(docs, self.embeddings)
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
                docs = text_splitter.split_text(data)
                self.db = FAISS.from_texts(docs, self.embeddings)
    
    def load_qa_model(self, tokenizer, model):
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, load_in_8bit_fp32_cpu_offload=True)
        model_4bit = AutoModelForCausalLM.from_pretrained(model, device_map="auto", quantization_config=quantization_config)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, padding=True, truncation=True, max_length=512)
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

        self.qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.db.as_retriever()
            )
    
    def respond_LLM(self, msg):
         self.response = self.qa({"query": msg})