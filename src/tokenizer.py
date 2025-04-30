from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# type hints in the sig
def build_bpe_tokenizer(vocab_size: int = 1000, min_frequency: int = 2, files: list[str] | None = None ) -> Tokenizer:
    """
    Train and return a Byte-Pair Encoding tokenizer.
    
    Args:
      vocab_size: maximum number of tokens in the BPE vocab.
      min_frequency: minimum frequency for a pair to be merged.
      files: list of paths to text files to train on (e.g. ["data/wiki.train.txt"]).
             If None, returns an untrained tokenizer.
    """
    # --- 1) Initialize the BPE model ---
    tokenizer = Tokenizer(models.BPE())

    # --- 2) Pre-tokenization: split on whitespace ---
    # Splits raw text on whitespace into initial “words” before BPE merges. 
    # Without it, BPE would operate on unsegmented character sequences, losing word boundaries.
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # --- 3) Trainer setup ---
    # - vocab_size: Bounding vocab_size prevents the tokenizer from creating millions of rare tokens. 
    #   An unlimited vocab leads to huge memory use, slower tokenization, and over-fitting on infrequent subwords.
    # - min_frequency: skip merges that occur too rarely.
    # - special_tokens: Ensures [PAD] is always in the final vocabulary, even if it never appears in your training files
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[PAD]"]
    )
    if files:
        tokenizer.train(files, trainer)
    
    # --- 4) Decoder & post-processor for ByteLevel BPE ---
    # These two lines set up how to turn token IDs back into strings, and
    # how to handle offsets/special‐token placement after encoding.
    tokenizer.decoder = decoders.ByteLevel()  # tells the tokenizer how to reconstruct UTF-8 text from token IDs.
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False) # applies the ByteLevel rules after tokenization (e.g. handling offsets, trimming spaces).

    # --- 5) Padding configuration ---
    # We reserved “[PAD]” above, so now fetch its ID and tell the tokenizer
    # to pad shorter sequences to the same length using this token. (6)
    pad_id = tokenizer.token_to_id("[PAD]")
    tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")  # instructs the tokenizer to right-pad shorter sequences with [PAD], crucial for batching inputs of variable length.
    
    return tokenizer
