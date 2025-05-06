import torch
import pickle
import numpy as np
import re
import random
import json
from model import BERT_Arch
import torch.serialization as ts
import sys
import json, random, torch, pickle
import spacy
from transformers import AutoModel, AutoConfig
from transformers import BertTokenizerFast, AutoModelForSequenceClassification
from transformers import AutoTokenizer
from langdetect import detect

# 1) Load refactored intents & lectors
with open('./content (1)/intents_refactored_cleaned_ukr.json', encoding='utf-8') as f:
    intents = json.load(f)['intents']
with open('./content (1)/lectors_ukr_translated.json', encoding='utf-8') as f:
    lectors = json.load(f)

# 2) Load the BERT intent model + tokenizer + label encoder
config = AutoConfig.from_pretrained('./content (1)/bert_finetuned')
bert_base = AutoModel.from_pretrained('./content (1)/bert_finetuned', config=config)

num_labels = 7
# 2) Reconstruct your full model
model = BERT_Arch(bert_base, num_labels=num_labels)
model.load_state_dict(torch.load('./content (1)/bert_finetuned/head_state.pt'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
with open('./content (1)/label_encoder.pkl','rb') as f:
    le = pickle.load(f)

# 3) Load the spaCy NER model
nlp_ner = spacy.load('./content (1)/fict_ner_model')
tokenizer = AutoTokenizer.from_pretrained('./content (1)/bert_finetuned')


# 4) Define helpers
def get_intent(text):
    toks = tokenizer(
        [text],
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids     = toks['input_ids'].to(device)
    attention_mask = toks['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    pred = logits.argmax(dim=1).item()
    return le.inverse_transform([pred])[0]

def extract_lector(text):
    doc = nlp_ner(text)
    for ent in doc.ents:
        return ent.text
    return None

def respond(user_msg):
    intent = get_intent(user_msg)
    print(f"Predicted intent: {intent}")
    if intent == 'lector_info':
        name = extract_lector(user_msg) or '<unknown>'
        print(f"Extracted LECTURER: {name}")
        key = name.lower().replace(' ','_')
        info = lectors.get(key)
        if info and info['responses']:
            return random.choice(info['responses'])
        else:
            return f"Sorry, I don't have data on {name}."
    # non-lecturer intents
    for it in intents:
        if it['tag'] == intent:
            return random.choice(it.get('responses', ["Sorry, I don't know."]))
    return "Sorry, I didn't understand that."