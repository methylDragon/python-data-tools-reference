# Modified from SUTD and https://github.com/bentrevett/pytorch-sentiment-analysis
# Sentiment Analysis on IMDB with FashionMNIST

# We're using packed sequences for training
# For more info: https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch

import torch.nn as nn
import torchtext
import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
import torch.optim as optim

import random
import time

# MODEL ========================================================================
class BidirectionalLSTM(nn.Module):
    def __init__(self,
                 input_dim,
                 embedding_dim=100,
                 hidden_dim=256,
                 output_dim=1,
                 n_layers=2,
                 bidirectional=True,
                 dropout=0.5,
                 pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(input_dim,
                                      embedding_dim,
                                      padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text)) # Map text to embedding

        # Pack sequence
        # Note: We move text_lengths to cpu due to a small bug
        # https://github.com/pytorch/pytorch/issues/43227
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu()
        )

        packed_output, (hidden, cell) = self.rnn(packed_embedded) # Feedforward

        # Unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        hidden = self.dropout(torch.cat((hidden[-2,:,:],
                                         hidden[-1,:,:]),
                                        dim = 1))

        return self.fc(hidden)


# TRAINING UTILITIES ===========================================================
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # Round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss, epoch_acc = 0, 0

    model.train() # Set to training mode

    for batch in iterator:
        optimizer.zero_grad()

        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)

        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss, epoch_acc = 0, 0

    model.eval() # Set to evaluation mode

    with torch.no_grad(): # Don't track gradients
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    # MAKE DETERMINISTIC =======================================================
    SEED = 1234
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # LOAD DATA ================================================================
    # Spacy is good for tokenisation in other languages
    TEXT = data.Field(tokenize = 'spacy', include_lengths = True)
    LABEL = data.LabelField(dtype = torch.float)

    # If slow, use this instead:
    # def tokenize(s):
    #     return s.split(' ')
    # TEXT = data.Field(tokenize=tokenize, include_lengths = True)

    # Test-valid-train split
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    train_data, valid_data = train_data.split(random_state = random.seed(SEED))

    # Visualise
    example = next(iter(test_data))
    example.label
    example.text

    # Note: Using glove embeddings (~900mb)
    TEXT.build_vocab(
        test_data,
        max_size = 25000,
        vectors = "glove.6B.100d",
        unk_init = torch.Tensor.normal_ # how to initialize unseen words not in glove
    )
    LABEL.build_vocab(test_data)

    # Data iterators
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size = 64,
        sort_within_batch = True,
        device = device)


    # MODEL ====================================================================
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] # Specifies index when word is missing

    model = BidirectionalLSTM(input_dim=len(TEXT.vocab),
                              embedding_dim=100,
                              hidden_dim=256,
                              output_dim=1,
                              n_layers=2, # To make LSTM deep
                              bidirectional=True,
                              dropout=0.5,
                              pad_idx=PAD_IDX)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    count_parameters(model) # 4,810,857 (wow!)

    # Copy embeddings to model
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    # Zero out <UNK> and <PAD> tokens
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    # TRAIN ====================================================================
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 5

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut2-model.pt')

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


    # WHEN DONE.. ==============================================================
    model.load_state_dict(torch.load('tut2-model.pt'))
    test_loss, test_acc = evaluate(model, test_iterator, criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

    # TRY WITH USER INPUT ======================================================
    import spacy
    nlp = spacy.load('en')

    def predict_sentiment(model, sentence):
        model.eval()
        tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        length = [len(indexed)]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(1)
        length_tensor = torch.LongTensor(length)
        prediction = torch.sigmoid(model(tensor, length_tensor))
        return prediction.item()

    predict_sentiment(model, "This film is great")
