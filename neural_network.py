import torch.nn as nn
import torch.optim as optim


class LSTMNetwork(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size):
        super(LSTMNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,num_layers=2,dropout=0.2)
        # The linear layer that maps from hidden state space to outputs
        self.output = nn.Linear(hidden_dim, output_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        outputs = self.output(lstm_out.view(len(sentence), -1))
        return outputs
        # return F.log_softmax(out_dim, dim=1)


class ModelTrainer:

    def __init__(self, model_params, training_data):
        self.model = LSTMNetwork(model_params.embedding_dim, model_params.hidden_dim, model_params.vocab_size,
                                 model_params.output_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=model_params.lr)
        self.train(model_params.epochs, training_data)

    def train(self, epochs, training_data):
        for epoch in range(0, epochs):  # again, normally you would NOT do 300 epochs, it is toy data
            running_loss = 0
            for sequences, targets in training_data:
                # clear gradients
                self.model.zero_grad()
                # Run our forward pass.
                predicted = self.model(sequences)
                # Compute the loss, gradients, and update the parameters by calling optimizer.step()
                loss = self.loss_function(predicted, targets)
                running_loss += loss
                loss.backward()
                self.optimizer.step()
            print("epoch {0} , run loss {1:.3f}".format(epoch, running_loss/len(training_data)))

    def get_model(self):
        return self.model
