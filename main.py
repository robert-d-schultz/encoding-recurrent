import torch
import torch.nn as nn
import random
import chardet

# This is a recurrent neural network that classifies documents by their encoding
# The input to the neural network is raw bytes
# Newswire data is used for training, this program reencodes it
# The model is compared to chardet's detect() function


# seed
seed = random.randint(1, 10000)
print("Random Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

# hyperparameters
n_epochs = 5000
lr = 0.0001
training_data = "./data/news.2009.en.shuffled.unique"

n_classes = 32

# these are encodings i picked out (latin characters, english, typically)
encodings_full = [ "ascii"
                 , "cp037"
                 , "cp437"
                 , "cp500"
                 , "cp850"
                 , "cp858"
                 , "cp1140"
                 , "cp1252"
                 , "latin_1"
                 , "iso2022_jp_2"
                 , "iso8859_15"
                 , "mac_roman"
                 , "utf_32"
                 , "utf_16"
                 , "utf_7"
                 , "utf_8"
                 ]
# these are the supported encodings for chardet
encodings_chardet = [ "Big5"
                    , "GB2312"
                    , "GB18030"
                    #, "EUC-TW" #python doesn't support
                    , "HZ-GB-2312"
                    #, "ISO-2022-CN" #python doesn't support
                    , "EUC-JP"
                    , "SHIFT_JIS"
                    , "ISO-2022-JP"
                    , "EUC-KR"
                    , "ISO-2022-KR"
                    , "KOI8-R"
                    , "MacCyrillic"
                    , "IBM855"
                    , "IBM866"
                    , "ISO-8859-5"
                    , "Windows-1251"
                    , "ISO-8859-2"
                    , "Windows-1250"
                    , "ISO-8859-5"
                    , "Windows-1251"
                    , "ISO-8859-1"
                    , "Windows-1252"
                    , "ISO-8859-7"
                    , "Windows-1253"
                    , "Windows-1254"
                    , "ISO-8859-8"
                    , "Windows-1255"
                    , "ISO-8859-9"
                    , "TIS-620"
                    , "UTF-32"
                    , "UTF-16"
                    , "utf-8"
                    , "ascii"
                    ]

def randomTrainingPair(f):
    string = f.readline()
    i = random.randint(0,n_classes-1)
    encoding = encodings_chardet[i]

    try:
        encoded = string.encode(encoding)

        bytes = [x for x in encoded]

        enc_tensor = torch.zeros(1, n_classes , dtype=torch.long)
        enc_tensor[0][i] = 1

        bytes_tensor = [torch.tensor([[b]], dtype=torch.float) for b in bytes]

        return torch.tensor([i], dtype=torch.long), bytes_tensor
    except UnicodeEncodeError:
        return randomTrainingPair(f) # pick a new encoding if the one selected doesn't work, kind of a hack


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(RNN, self).__init__()

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3


        self.i2h1 = nn.Linear(input_size + hidden_size3, hidden_size1)

        self.i2h2 = nn.Linear(hidden_size1, hidden_size2)

        self.i2h3 = nn.Linear(hidden_size2, hidden_size3)

        self.i2o = nn.Linear(hidden_size2, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden3):
        combined = torch.cat((input, hidden3), 1)
        hidden1 = self.i2h1(combined)
        hidden2 = self.i2h2(hidden1)
        hidden3 = self.i2h3(hidden2)
        output = self.i2o(hidden2)
        output = self.softmax(output)
        return output, hidden3

    def initHidden(self):
        return torch.zeros(1, self.hidden_size3)

n_input = 1
n_hidden1 = 256
n_hidden2 = 128
n_hidden3 = 64
n_encodings = n_classes
rnn = RNN(n_input, n_hidden1, n_hidden2, n_hidden3, n_encodings)
optimizer = torch.optim.SGD(rnn.parameters(), lr=lr)
criterion = nn.NLLLoss()

def train(enc_tensor, bytes_tensor):
    hidden3 = rnn.initHidden()

    rnn.zero_grad()

    for i in range(len(bytes_tensor)):
        output, hidden3 = rnn(bytes_tensor[i], hidden3)

    loss = criterion(output, enc_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-lr, p.grad.data)

    return output, loss.item()

def training():
    print("Training...")
    with open(training_data, mode="r", encoding="utf_8") as f:
        for epoch in range(0, n_epochs):
            enc_tensor, bytes_tensor = randomTrainingPair(f)
            output, loss = train(enc_tensor, bytes_tensor)
            if epoch % 50 == 0:
                print(str(epoch) + "/" + str(n_epochs) + "  Loss: " + str(loss))
        torch.save(rnn.state_dict(), "./out/model_output.pth")
        evaluation(f)

# evaluation set is just whatever follows the training set
def evaluation(f):
    print("Evaluating...")
    rnn_test = RNN(n_input, n_hidden1, n_hidden2, n_hidden3, n_encodings)
    rnn_test.load_state_dict(torch.load("./out/model_output.pth"))

    total = 0
    correct = 0
    correct_chardet = 0
    for line in f:

        string = f.readline()
        i = random.randint(0, n_classes-1)
        encoding = encodings_chardet[i]

        try:
            encoded = string.encode(encoding)

            bytes = [x for x in encoded]

            enc_tensor = torch.zeros(1, n_classes, dtype=torch.long)
            enc_tensor[0][i] = 1

            bytes_tensor = [torch.tensor([[b]], dtype=torch.float) for b in bytes]

            hidden3 = rnn.initHidden()
            for j in range(len(bytes_tensor)):
                output, hidden3 = rnn_test(bytes_tensor[j], hidden3)

            _, predicted_encoding = torch.max(output, 1)
            #print(encodings_chardet[predicted_encoding], encoding, chardet.detect(encoded)["encoding"])

            correct += (predicted_encoding.data.tolist()[0] == i)

            chardet_predicted = encodings_chardet.index(chardet.detect(encoded)["encoding"])
            correct_chardet += (chardet_predicted == i)

            total += 1

            #print(predicted_encoding.data.tolist()[0], i, encodings_chardet.index(chardet.detect(encoded)["encoding"]))

        except UnicodeEncodeError:
            pass
        if total >= 1000:
            break
    print("This model: " + str(correct/total) + "\tChardet: " + str(correct_chardet/total) + "\tChance: " + str(1/32))


# main
if __name__ == "__main__":
    training()
