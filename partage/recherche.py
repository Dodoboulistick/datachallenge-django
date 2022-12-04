from transformers import FlaubertTokenizer, FlaubertModel
import torch

import numpy as np

#analyzing word embeddings
from sklearn.neighbors import KDTree

#NLP
import string
from nltk import word_tokenize
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop


def load_flaubert(model):
    return FlaubertTokenizer.from_pretrained(model), FlaubertModel.from_pretrained(model)

## On transforme une phrase en vecteur grâce au model
def flaubert_word_embedding(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states[0][0].detach().numpy()

def flaubert_sentence_embedding(sentence, tokenizer, model):
    punctuations = list(string.punctuation) # liste de caractère de ponctuations
    sentence_bis = [i for i in word_tokenize(sentence) if i not in punctuations]
    # print(sentence_bis)
    sentence_vector = []
    for word in sentence_bis:
        sentence_vector.append(flaubert_word_embedding(word, tokenizer, model))
    res = np.mean(np.array(sentence_vector), axis=0)
    # print(len(res))
    return res

def flaubert_neighbours(content, embedded_contents, contents, tokenizer, model, n_neighbours, debug=False):
    # PRE-Processing the content, e.g the query
    punctuations = list(string.punctuation)
    stop_words_list = list(fr_stop)
    content_no_punctation = ''
    if debug: 
        for word in content:
            if word not in punctuations:
                content_no_punctation += word
    else : 
        for word in (content):
            if word not in punctuations:
                content_no_punctation += word
                
    tokenized_content = ''
    for word in content_no_punctation.split():
        if word.lower() not in stop_words_list:
            tokenized_content += word+' '
            
    zone = flaubert_sentence_embedding(tokenized_content, tokenizer, model)
    if debug:
        print(f'Looking for the {n_neighbours}th first neighbours of  : {content}')
    X = np.zeros((len(embedded_contents), zone.shape[0]))
    if debug:
        print(f"Searching for neighbours in {len(embedded_contents)} resources")
    
    zone = np.array(zone).reshape(1,zone.shape[0])
    tree = KDTree(embedded_contents,leaf_size=40)
    dist, ind = tree.query(zone, n_neighbours)
    res_array = []
    for k,indice in enumerate(ind[0]):
        if k>=0:
            res_array.append(indice)
            if debug:
                print(f"Neighbour n° {k+1}, index n° {indice}, distance = {dist[0][k]} : {contents[indice]['text']}, {contents[indice]['tag']}")
    return res_array, dist
