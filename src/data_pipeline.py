# src/data_pipeline.py
"""
Data pipeline for proof-of-concept Transformer training using local WikiText-2 splits.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# define how we want extract samples from the text (lines, padding, seq len, tokenizer)
class PoCDataset(Dataset):
    """
    Line-based text dataset: reads a text file with one sample per line.
    Pads or truncates token sequences to `seq_len`.

    In practice, you'd probably read the entire text, tokenize it once, 
    then split into fixed-length chunks (Eg. 256-1024 tokens), this ensures each training sample is same size, much less padding.
    """
    def __init__(self, filepath: str, tokenizer, seq_len: int):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f if line.strip()]
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_id = tokenizer.token_to_id("[PAD]")

    def __len__(self):
        return len(self.lines)

    # returns a fixed-length seq, padding or truncating if required
    def __getitem__(self, idx):
        enc = self.tokenizer.encode(self.lines[idx])
        ids = enc.ids[:self.seq_len]
        # pad to seq_len
        if len(ids) < self.seq_len:
            ids += [self.pad_id] * (self.seq_len - len(ids))
        
        input_ids = torch.tensor(ids, dtype=torch.long)

        # For labels, mask out pad tokens. The standard is to set them to be -100
        labels = input_ids.clone()
        labels[labels == self.pad_id] = -100

        return {
            "input_ids": input_ids,
            "labels":    labels,
        }


# create a dataloader for our specific PoCDataset
def make_dataloader(
    filepath: str,
    tokenizer,
    seq_len: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    distributed: bool = False,
    shuffle: bool | None = None
) -> DataLoader:
    """
    Creates a DataLoader for one split.
    If `distributed` is True, uses DistributedSampler.
    If `shuffle` is None, defaults to sampler is None.
    Otherwise uses the provided shuffle bool.
    """
    dataset = PoCDataset(filepath, tokenizer, seq_len)
    sampler = DistributedSampler(dataset) if distributed else None

    # decide shuffling: use passed arg if exists, otherwise 
    if shuffle is None:
        shuffle = (distributed is False)

    # shuffle if we're not using distributed mode, so that each epoch see data in a different random order
    # don't shuffle if we're in distributed mode, bc the DistributedSampler will handle that.
    # use pin_memory=True - allocates each bacth to pinned memory for faster access
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def get_train_val_loaders(
    train_path: str,
    val_path: str,
    tokenizer,
    seq_len: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    distributed: bool = False
) -> (DataLoader, DataLoader):
    """
    Returns (train_loader, val_loader) for local WikiText-2 splits.
    """
    train_loader = make_dataloader(
        filepath=train_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=distributed
    )

    # don't distribute or shuffle for validation set
    val_loader = make_dataloader(
        filepath=val_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=False,
        shuffle=False
    )
    return train_loader, val_loader
