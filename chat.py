import random
import json
import torch
from model import NeuralNet
from pythainlp import word_tokenize

# Check if GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents_th.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

FILE = "data_th.pth"
data = torch.load(FILE)

# Callback data
input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()  # Model evaluate

bot_name = "EmmyAi"
print("มาคุยกันเถอะ! พิมพ์ 'ออก' เพื่อจบการสนทนา")

def tokenize(sentence):
    return word_tokenize(sentence, keep_whitespace=False)

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [word for word in tokenized_sentence if word in all_words]
    bag = [1 if word in tokenized_sentence else 0 for word in all_words]
    return torch.tensor(bag, dtype=torch.float32).unsqueeze(0)

while True:
    sentence = input('ถาม: ')
    if sentence == "ออก":
        break

    # Tokenize the sentence
    tokenized_sentence = tokenize(sentence)
    print(f"Tokenized sentence: {tokenized_sentence}")

    # Create the bag of words
    X = bag_of_words(tokenized_sentence, all_words)
    print(f"Bag of words: {X}")

    X = X.to(device)

    # Model prediction
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    print(f"Predicted tag: {tag}")

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    print(f"Probability: {prob.item()}")

    if prob.item() > 0.5:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: ขอโทษค่ะ ฉันไม่เข้าใจ")
