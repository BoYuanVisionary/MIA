import torch
import torch.nn.functional as F
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import numpy as np


def load_model_and_tokenizer(model_path, tokenizer_path=None, max_memory=None, attn_implementation = None, device='cuda', **kwargs):

    model_args = {
        'trust_remote_code': True,
        'device_map': device,
        'torch_dtype': torch.bfloat16,  # https://github.com/huggingface/transformers/issues/25446 
        **kwargs
    }
    # It seems float16 does degrade the performance, but llama2 at least with an old version of torch (==1.10) doesn't support bfloat16
    
    if max_memory is not None:
        model_args['max_memory'] = max_memory

    # Differentiate based on the model name
    if 'gemma' not in model_path.lower() and attn_implementation is not None:
        model_args['attn_implementation'] = 'flash_attention_2'

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_args)

    # Load tokenizer
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False,
        use_cache=True,
    )
    # Use tokenizer.padding_side = 'left' for all decoder-only models
    tokenizer_path = tokenizer_path.lower()
    if 'oasst-sft-6-llama-30b' in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if 'guanaco' in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if 'llama' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'falcon' in tokenizer_path:
        tokenizer.padding_side = 'left'
    if 'gemma' in tokenizer_path:
        tokenizer.padding_side = 'left'
    if 'llama-3' in tokenizer_path:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def generate_batches(data, batch_size):
    """
    Generates batches from a list of lists.

    Args:
        data (list of lists): List containing lists of data to be batched.
        batch_size (int): Size of each batch.

    Yields:
        list of lists: A batch of the same structure as the input data.
    """
    num_samples = len(data[0])  # Assumes all inner lists have the same length
    for i in range(0, num_samples, batch_size):
        batch = [d[i:i + batch_size] for d in data] # Python's list slicing mechanism does not raise an error when the end index is out of bounds
        yield batch
        
