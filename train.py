import json
from pythainlp import word_tokenize
from nltk_utils import bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('intents_th.json', 'r', encoding='utf-8') as f:
   intents = json.load(f)

def tokenize(sentence):
   return word_tokenize(sentence, keep_whitespace=False)

def stem(word):
   return word

all_words = []
tags = []
xy = []
for intent in intents['intents']:
   tag = intent['tag']
   tags.append(tag)
   for pattern in intent['patterns']:
       w = tokenize(pattern)
       all_words.extend(w)
       xy.append((w, tag))

ignore_words = ['?', '!', '.', ',', 'ๆ', 'ฯ', 'ฯลฯ', '่', '้', '๊', '๋', '์']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
Y_train = []
for (pattern_sentence, tag) in xy:
   bag = bag_of_words(pattern_sentence, all_words)
   X_train.append(bag)

   label = tags.index(tag)
   Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

class ChatDataset(Dataset):
   def __init__(self):
       self.n_samples = len(X_train)
       self.x_data = X_train
       self.y_data = Y_train

   def __getitem__(self, index):
       return self.x_data[index], self.y_data[index]
   
   def __len__(self):
       return self.n_samples

# Hyperparameters  
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
   for (words, labels) in train_loader:
       words = words.to(device)
       labels = labels.to(device).long()

       # Forward pass
       outputs = model(words)
       loss = criterion(outputs, labels)

       # Backward pass and optimization
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       
   if (epoch+1) % 100 == 0:
       print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Final loss: {loss.item():.4f}')

data = {
   "model_state": model.state_dict(),
   "input_size": input_size,
   "output_size": output_size,
   "hidden_size": hidden_size,
   "all_words": all_words,
   "tags": tags
}

FILE = "data_th.pth"
torch.save(data, FILE)

print(f'Training complete. File saved to {FILE}')