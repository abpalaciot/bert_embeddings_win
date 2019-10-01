# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:07:35 2019

@author: XZ935UB
"""

import torch

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

def text_to_emb(text):
  marked_text = "[CLS] " + text + " [SEP]"
  tokenized_text = tokenizer.tokenize(marked_text)
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  segments_ids = [1] * len(tokenized_text)
  # Convert inputs to PyTorch tensors
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensors = torch.tensor([segments_ids])
  # Predict hidden states features for each layer
  with torch.no_grad():
      encoded_layers, _ = model(tokens_tensor, segments_tensors)
  # Convert the hidden state embeddings into single token vectors

  # Holds the list of 12 layer embeddings for each token
  # Will have the shape: [# tokens, # layers, # features]
  token_embeddings = [] 

  layer_i = 0
  batch_i = 0
  token_i = 0
  # For each token in the sentence...
  for token_i in range(len(tokenized_text)):
    
    # Holds 12 layers of hidden states for each token 
    hidden_layers = [] 
    
    # For each of the 12 layers...
    for layer_i in range(len(encoded_layers)):
      
      # Lookup the vector for `token_i` in `layer_i`
      vec = encoded_layers[layer_i][batch_i][token_i]
      
      hidden_layers.append(vec)
      
    token_embeddings.append(hidden_layers)

  # Stores the token vectors, with shape [22 x 768]
  token_vecs_sum = []

  # For each token in the sentence...
  for token in token_embeddings:
      # Sum the vectors from the last four layers.
      sum_vec = torch.sum(torch.stack(token)[-4:], 0)
      
      # Use `sum_vec` to represent `token`.
      token_vecs_sum.append(sum_vec)

  sentence_embedding = torch.mean(encoded_layers[11], 1)
  sentence_embedding =sentence_embedding.numpy()

  return(sentence_embedding)