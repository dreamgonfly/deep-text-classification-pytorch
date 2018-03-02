import torch
from torch.utils.data import Dataset, DataLoader

def pad_text(text, pad, min_length=None, max_length=None):
    length = len(text)
    if min_length is not None and length < min_length:
        return text + [pad]*(min_length - length)
    if max_length is not None and length > max_length:
        return text[:max_length]
    return text

class TextDataset(Dataset):
    
    def __init__(self, texts, dictionary, sort=False, min_length=None, max_length=None):

        PAD_IDX = dictionary.indexer(dictionary.PAD_TOKEN)
        
        self.texts = [([dictionary.indexer(token) for token in text], label) 
                          for text, label in texts]

        if min_length or max_length:
            self.texts = [(pad_text(text, PAD_IDX, min_length, max_length), label) 
                          for text, label in self.texts]

        if sort:
            self.texts = sorted(self.texts, key=lambda x: len(x[0]))
        
    def __getitem__(self, index):
        tokens, label = self.texts[index]
        return tokens, label
        
    def __len__(self):
        return len(self.texts)
    
class TextDataLoader(DataLoader):
    
    def __init__(self, dictionary, *args, **kwargs):
        super(TextDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.PAD_IDX = dictionary.indexer(dictionary.PAD_TOKEN)
    
    def _collate_fn(self, batch):
        text_lengths = [len(text) for text, label in batch]
        
        longest_length = max(text_lengths)

        texts_padded = [pad_text(text, pad=self.PAD_IDX, min_length=longest_length) for text, label in batch]
        labels = [label for text, label in batch]
        
        texts_tensor, labels_tensor = torch.LongTensor(texts_padded), torch.LongTensor(labels)
        return texts_tensor, labels_tensor