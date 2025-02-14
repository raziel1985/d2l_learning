from torch import nn

# 编码器
class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError

# 解码器
class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self, **kwargs).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def foward(self, X, state):
        raise NotImplementedError

# 合并编码器和解码器
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
