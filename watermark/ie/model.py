import torch
import torch.nn as nn
from transformers import AutoTokenizer
import torch.nn.functional as F



class Classifier(nn.Module):

    def __init__(self, input_dim, *args, **kwargs):

        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)   
        self.fc2 = nn.Linear(512, 256)        
        self.fc3 = nn.Linear(256, 128)        
        self.fc4 = nn.Linear(128, 64)         
        self.fc5 = nn.Linear(64, 32)          
        self.output = nn.Linear(32, 2)        
        self.relu = nn.LeakyReLU(negative_slope=0.01)  
        self.dropout = nn.Dropout(p=0.3)            

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.output(x)
        return x
    

class TokenTranslator:
    
    def __init__(self, original_tokenizer_path, proxy_tokenizer_path):
        self.original_tokenizer = AutoTokenizer.from_pretrained(original_tokenizer_path)
        self.proxy_tokenizer = AutoTokenizer.from_pretrained(proxy_tokenizer_path)

    def translate_via_ids(self, origin_input_ids):
        seq_len = origin_input_ids.shape[1]
        # Decode each sequence in the batch
        decoded_texts = [
            self.original_tokenizer.decode(seq, skip_special_tokens=True) 
            for seq in origin_input_ids
        ]
        
        # Encode each decoded text using the proxy tokenizer
        proxy_input_ids = self.proxy_tokenizer(
            decoded_texts,
            return_attention_mask=False,
            add_special_tokens=False
        )['input_ids']
        
        all_input_ids = []

        for input_ids in proxy_input_ids:
            s = input_ids[:seq_len]
            s = s + [self.proxy_tokenizer.pad_token_id] * (seq_len - len(s))
            all_input_ids.append(s)
        
        all_input_ids = torch.tensor(all_input_ids)
        
        return all_input_ids
    
class Unified_Feature_Translator:

    def __init__(self, original_tokenizer, proxy_tokenizer, embed_model):
        self.original_tokenizer = original_tokenizer
        self.proxy_tokenizer = proxy_tokenizer
        self.embed_model = embed_model

    def token_translator(self, origin_input_ids):
        decoded_text = self.original_tokenizer.decode(origin_input_ids, skip_special_tokens=True)
        embedding_tokens = self.proxy_tokenizer(decoded_text, return_tensors="pt", add_special_tokens=False)['input_ids'][0]
        return embedding_tokens
    
    def feature_extractor_next_token(self, input_ids, max_seq_len):
        batch_embedding_tokens = []
        for input_id in input_ids:
            embedding_tokens = self.token_translator(input_id)
            # padding
            if len(embedding_tokens) == 0:
                embedding_tokens = torch.tensor([self.proxy_tokenizer.pad_token_id])
            if len(embedding_tokens) > max_seq_len:
                embedding_tokens = embedding_tokens[-max_seq_len:]
            else:
                padding_length = max_seq_len - len(embedding_tokens)
                embedding_tokens = F.pad(embedding_tokens, (0, padding_length), value=self.proxy_tokenizer.pad_token_id)
            batch_embedding_tokens.append(embedding_tokens)
        batch_embedding_tokens = torch.stack(batch_embedding_tokens).to(self.embed_model.device)
        attention_mask = torch.ones_like(batch_embedding_tokens)
        attention_mask[batch_embedding_tokens == self.proxy_tokenizer.pad_token_id] = 0
        last_non_zero_idx = attention_mask.sum(dim=1) - 1
        with torch.no_grad():
            hidden_states = self.embed_model(input_ids=batch_embedding_tokens, attention_mask=attention_mask).last_hidden_state
        hidden_states = hidden_states[torch.arange(hidden_states.size(0)), last_non_zero_idx].cpu()
        return hidden_states
    
    def feature_extractor_for_bert(self, input_ids, max_seq_len, batch_size):
        num_tokens = len(input_ids)
        for idx in range(0, num_tokens, batch_size):
            batch_embedding_tokens = []
            for i in range(idx, min(idx + batch_size, num_tokens)):
                embedding_tokens = self.token_translator(input_ids[:i+1])
                if len(embedding_tokens) == 0:
                    embedding_tokens = torch.tensor([self.proxy_tokenizer.pad_token_id])
                if len(embedding_tokens) > max_seq_len:
                    embedding_tokens = embedding_tokens[-max_seq_len:]
                else:
                    padding_length = max_seq_len - len(embedding_tokens)
                    embedding_tokens = F.pad(embedding_tokens, (0, padding_length), value=self.proxy_tokenizer.pad_token_id)
                batch_embedding_tokens.append(embedding_tokens)
            batch_embedding_tokens = torch.stack(batch_embedding_tokens).to(self.embed_model.device)
            attention_mask = torch.ones_like(batch_embedding_tokens)
            attention_mask[batch_embedding_tokens == self.proxy_tokenizer.pad_token_id] = 0
            last_non_zero_idx = attention_mask.sum(dim=1) - 1
            with torch.no_grad():
                hidden_states = self.embed_model(input_ids=batch_embedding_tokens, attention_mask=attention_mask).last_hidden_state
            hidden_states = hidden_states[torch.arange(hidden_states.size(0)), last_non_zero_idx].cpu()
            yield hidden_states

    def feature_extractor_for_sentence_transformer(self, input_ids, max_seq_len, batch_size):
        num_tokens = len(input_ids)
        for idx in range(0, num_tokens, batch_size):
            batch_sentences = []
            for i in range(idx, min(idx + batch_size, num_tokens)):
                curr_tokens = input_ids[:i+1]
                embed_tokens = self.token_translator(curr_tokens)
                if embed_tokens.shape[0] > max_seq_len:
                    embed_tokens = embed_tokens[-max_seq_len:]
                sentence = self.proxy_tokenizer.decode(embed_tokens, skip_special_tokens=True)
                batch_sentences.append(sentence)
            embeddings = self.embed_model.encode(batch_sentences)
            yield embeddings, batch_sentences