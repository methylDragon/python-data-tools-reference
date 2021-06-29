# Modified from SUTD

import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim=100, hidden_dim=256, output_dim=1, n_layers=2,
                 bidirectional=True, dropout=0.5, pad_idx=0):
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
