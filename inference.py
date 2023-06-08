import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms
import string
import torch
import torch.nn as nn
import statistics
import torchvision.models as models
import torch.optim as optim
import tqdm
import pickle
from gtts import gTTS
from playsound import playsound


spacy_eng = spacy.load("en_core_web_sm")



# Load the dictionary from the file
with open("/home/rishabh/Pictures/image_captioning/vocab_itos.txt", "rb") as file:
    vocabulary= pickle.load(file)

# print('len of dict',len(vocabulary.keys()))

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        # self.times = []
        self.dropout = nn.Dropout(0.2)

    def forward(self, images):
        features = self.inception(images)
        
        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))

                predicted = output.argmax()

                result_caption.append(predicted.item())

                x = self.decoderRNN.embed(predicted).unsqueeze(0).unsqueeze(0)


                if vocabulary[predicted.item()] == "<EOS>":
                    break
                    
        return [vocabulary[idx] for idx in result_caption]


embed_size = 256
hidden_size = 256
vocab_size = 2970
num_layers = 1
learning_rate = 3e-4
num_epochs = 100


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device',device)

model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers)

# model.to(device)

model.load_state_dict(torch.load('/home/rishabh/Pictures/image_captioning/best.pth',map_location=device))

# model.eval()


def print_examples(model, device,vocabulary,img_path):

    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()

    test_img1 = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)

    print('shape',test_img1.shape)

    output=" ".join(model.caption_image(test_img1, vocabulary))

    return output


img_path='/home/rishabh/Pictures/image_captioning/1003163366_44323f5815.jpg'

if __name__=='__main__':
    output=print_examples(model,device,vocabulary,img_path)
    print('output',output[6:-5])
    speech = gTTS(output[6:-5], lang = 'en', slow = False)    
    speech.save('voice.mp3')
