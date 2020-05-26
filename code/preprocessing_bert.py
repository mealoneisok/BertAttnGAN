import pickle
import transformers
import config as cfg

dataroot = 'CUB\processed'

def to_bert_tokens(caps_text, tokenizer, max_len = None):
    caps_lst = []
    for caps in caps_text:
        cap_lst = []
        for cap in caps:
          cap = '[CLS] ' + cap
          cap = tokenizer.tokenize(cap)
          if max_len is not None:
              cap = cap[:max_len - 1]
              cap = cap + ['[SEP]']
              cap = cap + ['[PAD]' for _ in range(max_len - len(cap))]
          else:
              cap = cap + ['[SEP]']
          cap_lst.append(cap)
        caps_lst.append(cap_lst)
    return caps_lst

# make attention mask
def to_attention_mask(caps_text_token):
    def to_attention_mask_(tokens):
        attn_mask = []
        for token in tokens:
            if token == '[PAD]':
                attn_mask.append(0)
            else:
                attn_mask.append(1)
        return attn_mask

    attn_maskss = []
    for caps in caps_text_token:
        attn_masks = []
        for cap in caps:
            attn_masks.append(to_attention_mask_(cap))
        attn_maskss.append(attn_masks)
    return attn_maskss

#make token index
def to_bert_token_indices(caps_text, tokenizer):
    token_idxss = []
    for caps in caps_text:
        token_idxs = []
        for cap in caps:
          token_idxs.append(tokenizer.encode(cap, add_special_tokens=False))
        token_idxss.append(token_idxs)
    return token_idxss

if __name__ == '__main__':
    tokenizer = transformers.BertTokenizer.from_pretrained(cfg.BERT_PATH)

    with open(dataroot + '/caps_text_train.pkl', 'rb') as f:
        caps_text_train = pickle.load(f)
    with open(dataroot + '/caps_text_test.pkl', 'rb') as f:
        caps_text_test = pickle.load(f)
    
    caps_bert_train = to_bert_tokens(caps_text_train, tokenizer, max_len = cfg.MAX_LEN)
    caps_bert_test = to_bert_tokens(caps_text_test, tokenizer, max_len = cfg.MAX_LEN)
    attn_mask_train = to_attention_mask(caps_bert_train)
    attn_mask_test = to_attention_mask(caps_bert_test)
    token_bert_train = to_bert_token_indices(caps_bert_train, tokenizer)
    token_bert_test = to_bert_token_indices(caps_bert_test, tokenizer)

    with open(dataroot + '/attn_mask_train.pkl', 'wb') as f:
        pickle.dump(attn_mask_train, f)
    with open(dataroot + '/attn_mask_test.pkl', 'wb') as f:
        pickle.dump(attn_mask_test, f)

    with open(dataroot + '/token_bert_train.pkl', 'wb') as f:
        pickle.dump(token_bert_train, f)
    with open(dataroot + '/token_bert_test.pkl', 'wb') as f:
        pickle.dump(token_bert_test, f)

