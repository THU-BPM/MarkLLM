# ============================================
# transformers_config.py
# Description: Configuration for transformers
# ============================================

class TransformersConfig:
    """Configuration class for transformers."""

    def __init__(self, model, tokenizer, vocab_size=None, device='cuda', *args, **kwargs):
        """
            Initialize the transformers configuration.

            Parameters:
                model (object): The model object.
                tokenizer (object): The tokenizer object.
                vocab_size (int): The vocabulary size.
                device (str): The device to use.
                kwargs: Additional keyword arguments.
        """
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer) if vocab_size is None else vocab_size
        self.gen_kwargs = {}
        self.gen_kwargs.update(kwargs)