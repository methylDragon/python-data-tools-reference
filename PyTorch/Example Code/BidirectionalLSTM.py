import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim, #'100'
                           hidden_dim,
                           num_layers=n_layers, #set to two: makes our LSTM 'deep'
                           bidirectional=bidirectional, #bidirectional or not
                           dropout=dropout) #we add dropout for regularization

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout) #dropout layer

    def forward(self, text, text_lengths):

        embedded = self.dropout(self.embedding(text)) ## change the text to the embedding

        #pack sequence
        # note, we move text_lengths to cpu due to a small bug in current pytorch: https://github.com/pytorch/pytorch/issues/43227
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu()) #use packed padding

        packed_output, (hidden, cell) = self.rnn(packed_embedded) #feed to rnn

        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output) #unpack the padding

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)) #add dropout

        return self.fc(hidden)
