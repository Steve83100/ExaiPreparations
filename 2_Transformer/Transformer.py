import math
import torch
import torch.nn as nn
from torchinfo import summary


def sequence_mask(valid_lens, max_len):
        """
        Create padding mask from valid lengths.
        
        Args:
            valid_lens: shape (batch_size,), e.g., tensor([3, 5, 2])
            max_len: maximum sequence length (for padding)
        
        Returns:
            mask: shape (batch_size, max_len), True = padding (masked)
        """
        batch_size = valid_lens.shape[0]
        mask = torch.arange(max_len, device=valid_lens.device).unsqueeze(0).expand(batch_size, -1)
        # mask[i, j] = True if j >= valid_lens[i]
        return mask >= valid_lens.unsqueeze(1)


# ====================================================================


class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, embed_dim, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, embed_dim))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, embed_dim, 2, dtype=torch.float32) / embed_dim)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
    
# <Position>:
#
#     X (b:batch_size, n:seq_len, d:embed_dim)
#     |
#  -------
#  |     |
#  |     P (b, n, d)
#  |     |
#  -- + --
#     |
#  dropout
#     |
#    out (b, n, d)

# ====================================================================


class AddNorm(nn.Module):
    """The residual connection followed by layer normalization."""
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

# <AddNorm>:
#
#   X     Y (b, n, d)
#   |     |
#   |  dropout
#   |     |
#   -- + --
#      |
#  layernorm (on dimension d)
#      |
#     out (b, n, d)

# ====================================================================


class FFN(nn.Module):
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
    
# <FFN>:
#
#    X (b, n, d)
#    |
#  linear (b, n, d) -> (b, n, ffn_num_hiddens)
#    |
#   relu
#    |
#  linear (b, n, ffn_num_hiddens) -> (b, n, ffn_num_outputs)
#    |
#   out (b, n, ffn_num_outputs)

# ====================================================================


class EncoderBlock(nn.Module):
    """The Transformer encoder block."""
    def __init__(self, embed_dim, ffn_num_hiddens, num_heads, dropout, use_bias):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, use_bias, batch_first=True)
        self.addnorm1 = AddNorm(embed_dim, dropout)
        self.ffn = FFN(ffn_num_hiddens, embed_dim) # ffn_num_outputs = embed_dim, matching input X
        self.addnorm2 = AddNorm(embed_dim, dropout)

    def forward(self, X, valid_lens):
        in_len = X.shape[1]
        enc_mask = sequence_mask(valid_lens, in_len)
        Attn, self.attn_weights = self.attention(X, X, X, key_padding_mask=enc_mask)
        Y = self.addnorm1(X, Attn)
        return self.addnorm2(Y, self.ffn(Y))
    
# <EncBlock>:
#
#      X (b, in_n, d)
#      |
#  ---------
#  |       |
#  |    -------
#  |    |  |  |
#  |    q  k  v
#  |    |  |  |
#  |   attention (b, in_n, d) -> (b, in_n, d*h:num_heads) -> (b, in_n, d)
#  |       |
#  <AddNorm>
#      |
#      Y (b, in_n, d)
#      |
#  ---------
#  |       |
#  |     <FFN> (b, in_n, d) -> (b, in_n, ffn_num_hiddens) -> (b, in_n, d)
#  |       |
#  <AddNorm>
#      |
#     out (b, in_n, d)

# ====================================================================


class Encoder(nn.Module):
    """The Transformer encoder."""
    def __init__(self, input_vocab_size, embed_dim, ffn_num_hiddens, num_heads, num_layers, dropout, use_bias):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(input_vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("EncBlock"+str(i), EncoderBlock(embed_dim, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.embed_dim))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attn_weights
        return X

# <Encoder>:
#
#      X (b, in_n, v:vocab_size)
#      |
#    embed (b, in_n, v) -> (b, in_n, d)
#      |
#  <Position> (b, in_n, d)
#      |
#  <EncBlock> (b, in_n, d)
#      |
#     ... (total of num_layers blocks)
#      |
#  <EncBlock>
#      |
#     out (b, in_n, d)

# ====================================================================


class DecoderBlock(nn.Module):
    """The i-th block in the Transformer decoder."""
    def __init__(self, embed_dim, ffn_num_hiddens, num_heads, dropout, i, use_bias):
        super().__init__()
        self.i = i
        self.attention1 = nn.MultiheadAttention(embed_dim, num_heads, dropout, use_bias, batch_first=True)
        self.addnorm1 = AddNorm(embed_dim, dropout)
        self.attention2 = nn.MultiheadAttention(embed_dim, num_heads, dropout, use_bias, batch_first=True)
        self.addnorm2 = AddNorm(embed_dim, dropout)
        self.ffn = FFN(ffn_num_hiddens, embed_dim)
        self.addnorm3 = AddNorm(embed_dim, dropout)

    def forward(self, X, state, autoreg):
        # state[0] is EncBlock's output with size (b, n, d), used as K and V in enc-dec attention.
        # state[1] with size (b) specifies the valid length in each sequence in state[0].
        enc_outputs, enc_valid_lens = state[0], state[1]
        in_len = enc_outputs.shape[1]
        enc_mask = sequence_mask(enc_valid_lens, in_len)
        
        # With autoregression, model outputs token one at a time, so in each time step, an entire pass accross all blocks is done.
        # state[2] is the entire decoder's KV Cache, and state[2][i] is the cache for block i with size (b, 0~n, d).
        # Within one pass, in each block, state[2] is updated so that state[2][i] stores block i's input up to now.
        # Then, state[2][i] containing "all tokens up to now" is used as K and V during self-attention.
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        
        # Note that instead of self-regression, we use actual label as input duing teacher-forcing.
        # Therefore, output can be processed in parallel, and no KV Caching is needed.
        # But we still need to ensure that each time step only sees tokens up to now.
        # So we create attn_mask, which has size (n, n) and contains:
        # [
        #  [0, 1, 1, ..., 1],
        #  [0, 0, 1, ..., 1],
        #  ...
        #  [0, 0, 0, ..., 1]
        # ]
        # This will be identically broadcasted to each batch, masking out future tokens during self-attention.
        # For example, query 1 can only see the first token (itself); query 2 can see token 1 and 2.
        if autoreg:
            attn_mask = None
        else:
            num_steps = X.shape[1]
            attn_mask = torch.triu(
                torch.ones(num_steps, num_steps, dtype=torch.bool, device=X.device),
                diagonal=1
            )
            
        Attn1, self.self_attn_weights = self.attention1(X, key_values, key_values, attn_mask=attn_mask) # Self-attention
        Y = self.addnorm1(X, Attn1)
        Attn2, self.cross_attn_weights = self.attention2(Y, enc_outputs, enc_outputs, key_padding_mask=enc_mask) # Cross-attention
        Z = self.addnorm2(Y, Attn2)
        return self.addnorm3(Z, self.ffn(Z)), state
    
# <DecBlock>:
#
#      X (b, out_n, d)
#      |
#  ---------
#  |       |
#  |    -------
#  |    |  |  |
#  |    q  k  v
#  |    |  |  |
#  |   attention (b, out_n, d) -> (b, out_n, d*h:num_heads) -> (b, out_n, d)
#  |       |
#  <AddNorm>
#      |
#      Y (b, out_n, d)
#      |
#  ------     enc_outputs (b, in_n, d)
#  |    |     |
#  |    |  ----
#  |    |  |  |
#  |    q  k  v
#  |    |  |  |
#  |   attention (b, out_n, d) -> (b, out_n, d*h:num_heads) -> (b, out_n, d)
#  |       |
#  <AddNorm>
#      |
#      Y2 (b, out_n, d)
#      |
#  ---------
#  |       |
#  |     <FFN> (b, out_n, d) -> (b, out_n, ffn_num_hiddens) -> (b, out_n, d)
#  |       |
#  <AddNorm>
#      |
#     out (b, out_n, d)

# ====================================================================


class Decoder(nn.Module):
    """The Transformer encoder."""
    def __init__(self, output_vocab_size, embed_dim, ffn_num_hiddens, num_heads, num_layers, dropout, use_bias):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("DecBlock"+str(i), DecoderBlock(embed_dim, ffn_num_hiddens, num_heads, dropout, i, use_bias))
        self.dense = nn.LazyLinear(output_vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        """Given encoder outputs, initializes a state to be shared across layers."""
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state, autoreg):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.embed_dim))
        self.attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state, autoreg)
            self.attention_weights[0][i] = blk.self_attn_weights # Decoder self-attention weights
            self.attention_weights[1][i] = blk.cross_attn_weights # Encoder-decoder attention weights
        return self.dense(X), state
    
# <Decoder>:
#
#      X (b, out_n, v:vocab_size)
#      |
#    embed (b, out_n, v) -> (b, out_n, d)
#      |
#  <Position> (b, out_n, d)
#      |
#      |       enc_outputs (b, in_n, d)
#      |      /|
#      ------- |
#      |       |
#  <DecBlock>  |
#      |      /|
#      ------- |
#      |       |
#  <DecBlock>  |
#     ...     ...
#      |      /
#      -------
#      |
#  <DecBlock> (b, out_n, d)
#      |
#    dense (b, out_n, d) -> (b, out_n, v)
#      |
#     out (b, out_n, v)

# ====================================================================


class Transformer(nn.Module):
    """The Transformer model."""
    def __init__(self, input_vocab_size, output_vocab_size, embed_dim, ffn_num_hiddens, num_heads, num_layers, dropout, use_bias=False):
        super().__init__()
        print("\nCreating Transformer...")
        self.encoder = Encoder(input_vocab_size, embed_dim, ffn_num_hiddens, num_heads, num_layers, dropout, use_bias)
        self.decoder = Decoder(output_vocab_size, embed_dim, ffn_num_hiddens, num_heads, num_layers, dropout, use_bias)

    def forward(self, enc_X, dec_X, enc_valid_lens, out_len, autoreg=False):
        """
        enc_X have shape (batch_size, in_len, embed_dim), where in_len is the length of every input sequence.
        
        If autoreg = True, use auto-regression, which produces max_len tokens one by one, starting with dec_X.
        Uses only the newly-predicted token as decoder input, and save entire history in KV Cache.
        In this case, starting sequence dec_X has shape (batch_size, 1, embed_dim), and will output (batch_size, out_len, embed_dim).
        
        If autoreg = False, use teacher-forcing, which uses target sequence dec_X as decoder input, producing all tokens at once.
        In this case, target sequence dec_X has shape (batch_size, out_len, embed_dim), and so does output.
        """
        enc_outputs = self.encoder(enc_X, enc_valid_lens)
        state = self.decoder.init_state(enc_outputs, enc_valid_lens)
        
        if autoreg:
            # Store all outputs, not including the initial <SOS>
            outputs = []
            
            # Generate tokens one by one, and update dec_input
            dec_input = dec_X
            for _ in range(out_len):
                output, state = self.decoder(dec_input, state, True) # output: (batch_size, 1, vocab_size)
                outputs.append(output)
                predicted = output.argmax(dim=-1) # Greedily sample a token from distribution: (batch_size, 1)
                dec_input = predicted
            
            # Concatenate all outputs: List of (batch_size, 1, vocab_size) -> (batch_size, max_len, vocab_size)
            outputs = torch.cat(outputs, dim=1)
            return outputs
            
        else:
            return self.decoder(dec_X, state, False)[0]

# ====================================================================



if __name__ == "__main__":
    print("Testing...")
    
    # X = torch.rand((2, 100, 200)) # batchsize = 2, seqlen = 100, vocabsize = 200
    # valid_lens = torch.tensor([50, 20])
    # print(X.shape)

    # encoder_blk = EncoderBlock(200, 32, 8, 0.5)
    # Y = encoder_blk(X, valid_lens)
    # print(Y.shape)

    # decoder_blk = DecoderBlock(200, 32, 8, 0.5, 0)
    # state = [Y, valid_lens, [None]]
    # Z, new_state = decoder_blk(X, state)
    # print(Z.shape)
    
    # print(encoder_blk.attn_weights.shape)
    # print(decoder_blk.self_attn_weights.shape)
    # print(decoder_blk.cross_attn_weights.shape)
    
    # =================================================================
    
    BATCH_SIZE = 64
    INPUT_VOC = 200
    OUTPUT_VOC = 150
    IN_LEN = 100
    OUT_LEN = 90
    
    # In our complete transformer, we first go through nn.Embedding,
    # which expects word indexes as input, instead of one-hot vectors.
    # Therefore we only need 2D input (batchsize, seqlen), each value is a word index
    X = torch.randint(0, INPUT_VOC, (BATCH_SIZE, IN_LEN))
    valid_lens = torch.randint(10, 800, (BATCH_SIZE,))
    Y = torch.randint(0, OUTPUT_VOC, (BATCH_SIZE, OUT_LEN))
    start = torch.ones((BATCH_SIZE, 1), dtype=int)
    
    model = Transformer(
        input_vocab_size=200,
        output_vocab_size=150,
        embed_dim=32,
        ffn_num_hiddens=64,
        num_heads=8,
        num_layers=6,
        dropout=0.5
        )
    
    # print(model)
    print(X.shape)
    print(Y.shape)
    print(start.shape)
    print(model(X, Y, valid_lens, OUT_LEN, False).shape) # teacher-forcing
    print(model(X, start, valid_lens, OUT_LEN, True).shape) # auto-regression
    # summary(model, input_data = [X, Y, valid_lens])