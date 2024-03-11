import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import math
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

def build_and_plot_knowledge_graph_matplotlib(srl_results):
    G = nx.DiGraph()
    
    for result in srl_results:
        heads = result['head']
        types = result['type']
        tails = result['tail']
        
        for head in heads:
            for type in types:
                for tail in tails:
                    G.add_edge(head, type, label=tail)
    
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=2000, font_size=12, font_color="black", font_weight="bold", arrows=True)
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # Show plot
    plt.show()




# knowledge base class
class KB():
    def __init__(self):
        self.relations = []

    def are_relations_equal(self, r1, r2):
        return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

    def add_relation_small(self, r):
        if not self.exists_relation(r):
            self.relations.append(r)

    def print(self):
        print("Relations:")
        for r in self.relations:
            print(f"  {r}")

    def merge_relations(self, r1):
        r2 = [r for r in self.relations
              if self.are_relations_equal(r1, r)][0]
        spans_to_add = [span for span in r1["meta"]["spans"]
                        if span not in r2["meta"]["spans"]]
        r2["meta"]["spans"] += spans_to_add

    def add_relation(self, r):
        if not self.exists_relation(r):
            self.relations.append(r)
        else:
            self.merge_relations(r)

    def from_small_text_to_kb(self, text, verbose=False):
        # Tokenizer text
        model_inputs = tokenizer(text, max_length=512, padding=True, truncation=True,
                                return_tensors='pt')
        if verbose:
            print(f"Num tokens: {len(model_inputs['input_ids'][0])}")

        # Generate
        gen_kwargs = {
            "max_length": 216,
            "length_penalty": 0,
            "num_beams": 3,
            "num_return_sequences": 3
        }
        generated_tokens = model.generate(
            **model_inputs,
            **gen_kwargs,
        )
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

        # create kb
        for sentence_pred in decoded_preds:
            relations = self.extract_relations_from_model_output(sentence_pred)
            for r in relations:
                self.add_relation_small(r)

    # extract relations for each span and put them together in a knowledge base
    def from_text_to_kb(self, text, span_length=128, verbose=False):
        # tokenize whole text
        inputs = tokenizer([text], return_tensors="pt")

        # compute span boundaries
        num_tokens = len(inputs["input_ids"][0])
        if verbose:
            print(f"Input has {num_tokens} tokens")
        num_spans = math.ceil(num_tokens / span_length)
        if verbose:
            print(f"Input has {num_spans} spans")
        overlap = math.ceil((num_spans * span_length - num_tokens) / 
                            max(num_spans - 1, 1))
        spans_boundaries = []
        start = 0
        for i in range(num_spans):
            spans_boundaries.append([start + span_length * i,
                                    start + span_length * (i + 1)])
            start -= overlap
        if verbose:
            print(f"Span boundaries are {spans_boundaries}")

        # transform input with spans
        tensor_ids = [inputs["input_ids"][0][boundary[0]:boundary[1]]
                    for boundary in spans_boundaries]
        tensor_masks = [inputs["attention_mask"][0][boundary[0]:boundary[1]]
                        for boundary in spans_boundaries]
        inputs = {
            "input_ids": torch.stack(tensor_ids),
            "attention_mask": torch.stack(tensor_masks)
        }

        # generate relations
        num_return_sequences = 3
        gen_kwargs = {
            "max_length": 256,
            "length_penalty": 0,
            "num_beams": 3,
            "num_return_sequences": num_return_sequences
        }
        generated_tokens = model.generate(
            **inputs,
            **gen_kwargs,
        )

        # decode relations
        decoded_preds = tokenizer.batch_decode(generated_tokens,
                                            skip_special_tokens=False)

        i = 0
        for sentence_pred in decoded_preds:
            current_span_index = i // num_return_sequences
            relations = self.extract_relations_from_model_output(sentence_pred)
            for relation in relations:
                relation["meta"] = {
                    "spans": [spans_boundaries[current_span_index]]
                }
                self.add_relation(relation)
            i += 1
    # from https://huggingface.co/Babelscape/rebel-large
    def extract_relations_from_model_output(self, text):
        relations = []
        relation, subject, relation, object_ = '', '', '', ''
        text = text.strip()
        current = 'x'
        text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
        for token in text_replaced.split():
            if token == "<triplet>":
                current = 't'
                if relation != '':
                    relations.append({
                        'head': subject.strip(),
                        'type': relation.strip(),
                        'tail': object_.strip()
                    })
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    relations.append({
                        'head': subject.strip(),
                        'type': relation.strip(),
                        'tail': object_.strip()
                    })
                object_ = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        if subject != '' and relation != '' and object_ != '':
            relations.append({
                'head': subject.strip(),
                'type': relation.strip(),
                'tail': object_.strip()
            })
        return relations
df = pd.read_csv('web_search_results.csv') 
  
# converting ;body' column into text paragraphs
text = ''
kb = KB()
for i in df['body']:
    # kb.from_small_text_to_kb(i)
    kb.from_text_to_kb(i)
kb.print()
build_and_plot_knowledge_graph_matplotlib(kb.relations)