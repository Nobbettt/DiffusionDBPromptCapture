from torch import nn

class AttentionNetwork(nn.Module):
    def __init__(self, encoder_dimemsion, decoder_dimension, attention_dimension):
        super(AttentionNetwork, self).__init__()
        self.encoder_attention = nn.Linear(encoder_dimemsion, attention_dimension)
        self.decoder_attention = nn.Linear(decoder_dimension, attention_dimension) 
        self.full_attention = nn.Linear(attention_dimension, 1) 
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, encoder_out, decoder_hidden):
        attention_from_encoder = self.encoder_attention(encoder_out) 
        attention_from_decoder = self.decoder_attention(decoder_hidden) 
        attention_combined = self.full_attention(self.relu(attention_from_encoder + attention_from_decoder.unsqueeze(1))).squeeze(2) 
        alpha = self.softmax(attention_combined)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return attention_weighted_encoding, alpha