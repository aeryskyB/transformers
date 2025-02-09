import jax.numpy as jnp
from jax import random
from positional_encoder import Sinusoidal
from encoder import Encoder
from decoder import Decoder

class Transformer:
    def __init__(self, key=random.key(0), d_model=512, stack_size=6, dtype=jnp.float32):
        keys = random.split(key, 3)
        self.encoder = Encoder(key=keys[0], stack_size=stack_size, dtype=dtype)
        self.decoder = Decoder(key=keys[1], stack_size=stack_size, dtype=dtype)
        self.pos_encoder = Sinusoidal(d_model=d_model)

    def __call__(self, input_embed, output_embed):
        input_embed_pe = input_embed + self.pos_encoder(n=input_embed.shape[0])
        output_embed_pe = output_embed + self.pos_encoder(n=output_embed.shape[0])
        y_enc = self.encoder(input_embed_pe)
        y_dec = self.decoder(output_embed_pe, y_enc)

        """
        I won't use the final linear-transformation + softmax
        because this linear-transformation is learned and
        it shares weight with input and output embedding layers
        ref. #3.4 @[Attention Is All You Need]
        """

        return y_dec


