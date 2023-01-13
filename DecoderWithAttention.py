import torch
from torch import nn
from AttentionNetwork import AttentionNetwork

class DecoderWithAttention(nn.Module):

    def __init__(self, attention_dimension, embedding_dimension, decoder_dimension, vocab_size, device = torch.device("cpu"), encoder_dimension=2048, dropoutFraction=0.5):
        super(DecoderWithAttention, self).__init__()
        
        self.device = device
        self.vocab_size = vocab_size

        self.attention = AttentionNetwork(encoder_dimension, decoder_dimension, attention_dimension) 
        self.embedding = nn.Embedding(vocab_size, embedding_dimension) 
        self.dropout = nn.Dropout(p=dropoutFraction)
        self.decode_step = nn.LSTMCell(embedding_dimension + encoder_dimension, decoder_dimension, bias=True)  
        self.init_h = nn.Linear(encoder_dimension, decoder_dimension) 
        self.init_c = nn.Linear(encoder_dimension, decoder_dimension) 
        self.f_beta = nn.Linear(decoder_dimension, encoder_dimension)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dimension, vocab_size) 
        self.init_weights() 


    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)


    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)


    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune


    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out) 
        c = self.init_c(mean_encoder_out)
        return h, c


    def forward(self, encoder_out, encoded_captions, caption_lengths):  
        batch_size = encoder_out.size(0)
        encoder_dimension = encoder_out.size(-1)

        encoder_out = encoder_out.view(batch_size, -1, encoder_dimension) 
        num_pixels = encoder_out.size(1)

        embeddings = self.embedding(encoded_captions) 

        h, c = self.init_hidden_state(encoder_out) 

        decode_lengths = (caption_lengths-1).tolist()

        
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(self.device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
     
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
                                                
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))

            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha


        return predictions, encoded_captions, decode_lengths, alphas