from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)


def tokenize_and_chunk(samples, tokenizer, seq_len, field):
    text = "\n\n".join(samples[field])
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    num_chunks = input_ids.numel() // seq_len
    chunks = input_ids[: num_chunks * seq_len].view(num_chunks, seq_len)
    return IndexDataset(chunks)


def get_test_data(name, tokenizer, seq_len=2048, batch_size=4):
    if "wikitext2" in name:
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        dataset = tokenize_and_chunk(data, tokenizer, seq_len, "text")

    elif "ptb" in name:
        data = load_dataset("ptb_text_only", "penn_treebank", split="test")
        dataset = tokenize_and_chunk(data, tokenizer, seq_len, "sentence")

    elif "c4" in name:
        data = load_dataset("json", data_files="utils/c4-validation.json")["train"]
        dataset = tokenize_and_chunk(data[:2000], tokenizer, seq_len, "text")

    else:
        raise ValueError(f"Unsupported dataset name: {name}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
