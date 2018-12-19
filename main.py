import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
import random
import chardet
import pickle
import sklearn.metrics

import sys

# This is a recurrent neural network that classifies documents by their encoding
# The input to the neural network is raw bytes
# Newswire data is used for training, this program reencodes it
# The model is compared to chardet's detect() function

# Seed
seed = random.randint(1, 10000)
print("Random Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

# Gpu stuff
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Parameters
encodings = [ "Windows-1252"
            , "utf-8"
            #, "UTF-16"
            #, "UTF-32"
            ]
n_classes = len(encodings)

batch_size = 100
n_epochs = 1

input_size = 3
hidden_size = 32
layer_n = 1
output_size = n_classes

lr = 0.0001
criterion = nn.NLLLoss()

training_data = "./data/news.2009.en.shuffled.unique"

# Creates training and evauation sets
def preprocess_data():
    with open(training_data, mode="r", encoding="utf_8") as f:

        # Training set
        print("Building training set...")
        train_n = 900000
        train_out = []
        for line in f:
            string = f.readline()

            # Strip newline character from end
            string = string.strip()

            enc_strings = []
            labels = []
            for i in range(n_classes):
                try:
                    enc_string = string.encode(encodings[i])
                    enc_strings.append(enc_string)
                    labels.append(i)
                except UnicodeEncodeError:
                    continue

            # If the string encodes to the same bytes for any two classes, then its not useful for training
            if len(set(enc_strings)) < len(enc_strings):
                continue
            else:
                for enc_string, label in zip(enc_strings, labels):
                    print(string)
                    print(enc_strings)
                    bytes = [x for x in enc_string]

                    # bigrams, or maybe "bi-bytes"?
                    bytes_tensor = list(zip(*[bytes[i:] for i in range(input_size)]))

                    bytes_tensor = torch.tensor(bytes_tensor, dtype=torch.float).cuda()

                    enc_tensor = torch.tensor(label, dtype=torch.long).cuda()

                    train_out.append((bytes_tensor, enc_tensor, []))
                    if len(train_out) % 1000 == 0:
                        print(str(len(train_out)) + "/" + str(train_n))

            if train_n <= len(train_out):
                break

        # Evaluation set
        print("Building evaluation set...")
        eval_n = 100000
        eval_out = []
        for line in f:
            string = f.readline()

            string = string.strip()

            enc_strings = []
            labels = []
            for i in range(n_classes):
                try:
                    enc_string = string.encode(encodings[i])
                    enc_strings.append(enc_string)
                    labels.append(i)
                except UnicodeEncodeError:
                    continue

            if len(set(enc_strings)) < len(enc_strings):
                continue
            else:
                for enc_string, label in zip(enc_strings, labels):
                    bytes = [x for x in enc_string]

                    bytes_tensor = list(zip(*[bytes[i:] for i in range(input_size)]))

                    bytes_tensor = torch.tensor(bytes_tensor, dtype=torch.float).cuda()

                    enc_tensor = torch.tensor(label, dtype=torch.long).cuda()

                    eval_out.append((bytes_tensor, enc_tensor, enc_string))
                    if len(eval_out) % 1000 == 0:
                        print(str(len(eval_out)) + "/" + str(eval_n))

            if eval_n <= len(eval_out):
                break

    random.shuffle(train_out)
    random.shuffle(eval_out)
    with open('cache/training_data.pickle','wb') as g:
        pickle.dump(train_out, g)
    with open('cache/evaluation_data.pickle','wb') as h:
        pickle.dump(eval_out, h)



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layer_n, output_size, ngpu):
        super(LSTM, self).__init__()
        self.ngpu = ngpu

        self.hidden_size = hidden_size
        self.layer_n = layer_n

        self.lstm = nn.LSTM(input_size, hidden_size, layer_n, batch_first=True)

        self.linear = nn.Sequential(
                    nn.Linear(hidden_size, output_size),
                    #nn.Dropout(0.1),
                    nn.LogSoftmax(dim=1)
                    )

    def forward(self, x):
        h0 = Variable(torch.zeros(self.layer_n, x.batch_sizes[0].item(), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.layer_n, x.batch_sizes[0].item(), self.hidden_size)).cuda()

        out, (hn, cn) = self.lstm(x, (h0, c0))

        #print(out)
        data = out.data
        out_ = PackedSequence(self.linear(data), out.batch_sizes)

        return self.linear(hn[-1]), out_


# This pads batches so they can be packaged into PackedSequence's in DataLoader's
# I have no idea why this isn't a standard function
def pad_packed_collate(batch):
    if len(batch) == 1:
        sigs, labels, enc_strings = batch[0][0], batch[0][1], batch[0][2]
        lengths = [sigs.size(0)]
        sigs.unsqueeze_(0)
        labels.unsqueeze_(0)
        enc_strings = [enc_strings]
    if len(batch) > 1:
        sigs, labels, enc_strings, lengths = zip(*[(a, b, c, a.size(0)) for (a,b,c) in sorted(batch, key=lambda x: x[0].size(0), reverse=True)])
        max_len = sigs[0].size(0)
        sigs = [torch.cat((s, torch.zeros((max_len - s.size(0), input_size), dtype=torch.float).cuda()), dim=0) if s.size(0) != max_len else s for s in sigs]
        sigs = torch.stack(sigs, 0)
        labels = torch.stack(labels, 0)
    packed_batch = pack_padded_sequence(Variable(sigs).cuda(), list(lengths), batch_first=True)
    return packed_batch, labels, list(enc_strings)


model = LSTM(input_size, hidden_size, layer_n, output_size, ngpu).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

# Training
def training():

    with open('cache/training_data.pickle','rb') as g:
        t_data = pickle.load(g)

    train_loader = torch.utils.data.DataLoader(t_data, batch_size=batch_size, collate_fn=pad_packed_collate, shuffle=True, drop_last=True)

    print("Training...")
    for epoch in range(0, n_epochs):
        for n, (bytes_tensor, enc_tensor, _) in enumerate(train_loader):

            optimizer.zero_grad()

            outputs, _ = model(bytes_tensor)
            loss = criterion(outputs, enc_tensor)
            loss.backward()

            optimizer.step()

            print(str(epoch+1) + "/" + str(n_epochs) + "  " + str(n * batch_size) + "/" + str(len(t_data)) + "  Loss: " + str(round(loss.item(),5)))
        print("Saving...")
        torch.save(model.state_dict(), "./out/model_output.pth")


# Evaluation on model
def evaluation_model():
    with open('cache/evaluation_data.pickle','rb') as h:
        e_data = pickle.load(h)

    eval_loader = torch.utils.data.DataLoader(e_data, batch_size=batch_size, collate_fn=pad_packed_collate, shuffle=False)

    print("Evaluating model...")
    test_model = LSTM(input_size, hidden_size, layer_n, output_size, ngpu).to(device)
    test_model.load_state_dict(torch.load("./out/model_output.pth"))
    test_model.eval()

    model_predictions = []
    chardet_predictions = []
    labels = []
    with torch.no_grad():
        for n, (bytes_tensor, enc_tensor, enc_string) in enumerate(eval_loader):

            outputs, _ = test_model(bytes_tensor)

            label = list(enc_tensor.detach().cpu().numpy())
            labels.extend(enc_tensor)

            model_prediction = list(torch.max(outputs, 1)[1].detach().cpu().numpy())
            model_predictions.extend(model_prediction)

            if n % 100 == 0:
                print(str(n * batch_size) + "/" + str(len(e_data)))

    print("Model Accuracy: " + str(sklearn.metrics.accuracy_score(labels, model_predictions)))
    print("Model Precision: " + str(sklearn.metrics.precision_score(labels, model_predictions)))
    print("Model Recall: " + str(sklearn.metrics.recall_score(labels, model_predictions)))
    print("Model F1 Score: " + str(sklearn.metrics.f1_score(labels, model_predictions)))
    print("Model Confusion Matrix: " + str(sklearn.metrics.confusion_matrix(labels, model_predictions)))

# Evaluation on chardet
def evaluation_chardet():
    with open('cache/evaluation_data.pickle','rb') as h:
        e_data = pickle.load(h)

    print("Evaluating chardet...")
    chardet_predictions = []
    labels = []
    for n, (bytes_tensor, enc_tensor, enc_string) in enumerate(e_data):

        label = enc_tensor.detach().cpu().numpy()
        labels.append(label)

        # Get chardet version 4.0.0 if this doesn't work
        chardet_prediction = chardet.detect_all(enc_string)

        # Only compare the UTF-8 and Windows-1252 predictions
        # This skips bugs in chardet's Korean CP949 and Turkish Windows-1254/ISO-8859-9 detectors
        utf8 = 0
        windows1252 = 0
        for e in chardet_prediction:
            if e["encoding"] == "utf-8":
                utf8 = e["confidence"]

            # ISO-8859-1 is a subset of Windows-1252, treat them the same
            elif (e["encoding"] in ["Windows-1252", "ISO-8859-1"]) and e["confidence"] > windows1252:
                windows1252 = e["confidence"]

        if utf8 > windows1252:
            chardet_prediction_ = 1
        elif utf8 < windows1252:
            chardet_prediction_ = 0
        # If chardet detects neither..., default to Windows-1252 for output purposes
        else:
            print("Tie in chardet prediction")
            print(label, chardet_prediction, enc_string)
            chardet_prediction_ = 0

        if label == 0 and chardet_prediction_ == 1:
            print("label: ", label, "predicted: ", chardet_prediction_, enc_string)

        chardet_predictions.append(chardet_prediction_)

        if n % 1000 == 0:
            print(str(n) + "/" + str(len(e_data)))

    print("Chardet Accuracy: " + str(sklearn.metrics.accuracy_score(labels, chardet_predictions)))
    print("Chardet Precision: " + str(sklearn.metrics.precision_score(labels, chardet_predictions)))
    print("Chardet Recall: " + str(sklearn.metrics.recall_score(labels, chardet_predictions)))
    print("Chardet F1 Score: " + str(sklearn.metrics.f1_score(labels, chardet_predictions)))
    print("Chardet Confusion Matrix: " + str(sklearn.metrics.confusion_matrix(labels, chardet_predictions)))


# main
if __name__ == "__main__":
    preprocess_data()
    training()
    evaluation_model()
    evaluation_chardet()
