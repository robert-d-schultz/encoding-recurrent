import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
import random
import chardet
import pickle

# This is a recurrent neural network that classifies documents by their encoding
# The input to the neural network is raw bytes
# Newswire data is used for training, this program reencodes it
# The model is compared to chardet's detect() function


# seed
seed = random.randint(1, 10000)
print("Random Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

n_classes = 4

encodings = [ "ascii"
            , "ISO-8859-1"
            , "Windows-1252"
            , "utf-8"
            ]

# These are the top encodings used for websites
encodings_top = [ "utf-8"
                , "ISO-8859-1"
                , "Windows-1251"
                , "Windows-1252"
                , "SHIFT_JIS"
                , "GB2312"
                , "EUC-KR"
                , "ISO-8859-2"
                # , "GBK" #chardet doesn't support?
                , "Windows-1250"
                , "EUC-JP"
                , "Big5"
                #, "ISO-8859-15" #chardet doesn't support?
                #, "Windows-1256" #chardet doesn't support?
                , "ISO-8859-9"
                , "Windows-1254"
                ]

# These are the supported encodings for chardet
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

training_data = "./data/news.2009.en.shuffled.unique"

def preprocess_data():
    with open(training_data, mode="r", encoding="utf_8") as f:
        train_n = 1000
        ns = [0, 0, 0, 0]
        train_out = []
        for line in f:
            if(ns == [train_n, train_n, train_n, train_n]):
                break
            #print("New line.")
            string = f.readline()
            for i in range(0, n_classes):
                try:
                    #print(str(ns))
                    #print("Attempting to encode as " + str(i) + "...")

                    encoded = string.encode(encodings[i])

                    bytes = [x for x in encoded]

                    enc_tensor = [0]*n_classes
                    enc_tensor[i] = 1
                    enc_tensor = torch.tensor(enc_tensor, dtype=torch.long)

                    bytes_tensor = [[b] for b in bytes]
                    bytes_tensor = torch.tensor(bytes_tensor, dtype=torch.float)

                    if (ns[i] >= train_n):
                        #print("Too much " + str(i) + ", skipping.")
                        break

                    if (ns[i] % 50 == 0):
                        print(str(ns[i]) + "/" + str(train_n) + " for " + str(i))

                    #print("Success")
                    ns[i] += 1
                    train_out.append((bytes_tensor, enc_tensor))
                    break
                except (UnicodeEncodeError):
                    #print("Failed")
                    continue

        eval_n = 1000
        eval_out = []
        for line in f:
            string = f.readline()
            for i in range(0, n_classes):
                try:
                    #print("Attempting to encode as " + str(i) + "...")

                    encoded = string.encode(encodings[i])

                    bytes = [x for x in encoded]

                    enc_tensor = [0]*n_classes
                    enc_tensor[i] = 1
                    enc_tensor = torch.tensor(enc_tensor, dtype=torch.long)

                    bytes_tensor = [[b] for b in bytes]
                    bytes_tensor = torch.tensor(bytes_tensor, dtype=torch.float)

                    #print("Success")
                    eval_n -= 1
                    eval_out.append((bytes_tensor, enc_tensor, i, encoded))
                    break
                except (UnicodeEncodeError):
                    #print("Failed")
                    continue
            if(eval_n == 0):
                break

    random.shuffle(train_out)
    random.shuffle(eval_out)
    with open('cache/training_data.pickle','wb') as g:
        pickle.dump(train_out, g)
    with open('cache/evaluation_data.pickle','wb') as h:
        pickle.dump(eval_out, h)



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, layer_n, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.layer_n = layer_n

        self.rnn = nn.RNN(input_size, hidden_size, layer_n, batch_first=True,
                          nonlinearity='relu')

        self.fc = nn.Sequential(
                    nn.Linear(hidden_size, output_size),
                    nn.Dropout(0.1),
                    nn.LogSoftmax(dim=1)
                    )

    def forward(self, x, batch_size):
        h0 = Variable(torch.zeros(self.layer_n, batch_size, self.hidden_size))

        if (type(x) == PackedSequence):
            out, hn = self.rnn(x, h0)
            data = out.data
            out = PackedSequence(self.fc(data), out.batch_sizes)
        else:
            out, hn = self.rnn(x, h0)
            out = self.fc(out)
        return out


batch_size = 50
n_epochs = 5

def pad_packed_collate(batch):
    if len(batch) == 1:
        sigs, labels = batch[0][0], batch[0][1]
        #sigs = sigs.t()
        lengths = [sigs.size(0)]
        sigs.unsqueeze_(0)
        labels.unsqueeze_(0)
    if len(batch) > 1:
        sigs, labels, lengths = zip(*[(a, b, a.size(0)) for (a,b) in sorted(batch, key=lambda x: x[0].size(0), reverse=True)])
        max_len = sigs[0].size(0)
        sigs = [torch.cat((s, torch.zeros((max_len - s.size(0), 1), dtype=torch.float)), dim=0) if s.size(0) != max_len else s for s in sigs]
        sigs = torch.stack(sigs, 0)
        labels = torch.stack(labels, 0)
    packed_batch = pack_padded_sequence(Variable(sigs), list(lengths), batch_first=True)
    return packed_batch, labels


input_size = 1
hidden_size = 128
layer_n = 3
output_size = n_classes
model = RNN(input_size, hidden_size, layer_n, output_size)


lr = 0.001
criterion = nn.NLLLoss()
#criterion = nn.CrossEntropyLoss()

#optimizer = torch.optim.SGD(rnn.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))


# Training
def training():

    with open('cache/training_data.pickle','rb') as g:
        t_data = pickle.load(g)

    train_loader = torch.utils.data.DataLoader(t_data, batch_size=batch_size, collate_fn=pad_packed_collate, shuffle = False)

    print("Training...")
    for epoch in range(0, n_epochs):
        for n, (bytes_tensor, enc_tensor) in enumerate(train_loader):

            optimizer.zero_grad()

            packed_output = model(bytes_tensor, batch_size)
            output, _ = pad_packed_sequence(packed_output)

            loss = criterion(output[-1], torch.max(enc_tensor, 1)[1])
            loss.backward()

            optimizer.step()

            print(str(epoch+1)+"/"+str(n_epochs) + "  " + str(n * batch_size)+"/"+str(len(t_data)) + "  Loss: " + str(round(loss.item(),5)))
        print("Saving...")
        torch.save(model.state_dict(), "./out/model_output.pth")

# Evaluation
def evaluation():
    with open('cache/evaluation_data.pickle','rb') as h:
        e_data = pickle.load(h)

    eval_loader = torch.utils.data.DataLoader(e_data, batch_size=1, shuffle = False)

    print("Evaluating...")
    test_model = RNN(input_size, hidden_size, layer_n, output_size)
    test_model.load_state_dict(torch.load("./out/model_output.pth"))

    total = 0
    correct = 0
    correct_chardet = 0

    for n, (bytes_tensor, enc_tensor, i, encoded) in enumerate(eval_loader):

        output = test_model(bytes_tensor, 1)
        _, predicted_encoding = torch.max(output[0][-1].data, 0)

        correct += (predicted_encoding == i).item()

        try:
            chardet_predicted = encodings.index(chardet.detect(encoded[0])["encoding"])
        except (ValueError):
            chardet_predicted = -1
        correct_chardet += (chardet_predicted == i).item()

        total += 1

        print(i.item(), predicted_encoding.item(), chardet_predicted)

    print("This model: " + str(round(correct/total,5)) + "\tChardet: " + str(round(correct_chardet/total,5)))


# main
if __name__ == "__main__":
    #preprocess_data()
    training()
    evaluation()
