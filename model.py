import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)


class CodeGPTSensor(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeGPTSensor, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.classifier = nn.Linear(config.hidden_size, 2)
    
    def get_xcode_vec(self, source_ids):
        mask = source_ids.ne(self.config.pad_token_id)
        out = self.encoder(source_ids, attention_mask=mask.unsqueeze(1) * mask.unsqueeze(2),output_hidden_states=True)

        token_embeddings = out[0]

        sentence_embeddings = (token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)  # averege
        sentence_embeddings = sentence_embeddings

        return sentence_embeddings

    def forward(self, input_ids, contrast_ids=None, labels=None):
        
        vec = self.get_xcode_vec(input_ids)
        logits = self.classifier(vec)
        prob = nn.functional.softmax(logits, dim=-1)
        
        if labels is not None:
            # cross entropy loss for classifier
            loss = F.cross_entropy(logits, labels)
            return loss, prob
        else:
            return prob
    
    def predict(self, code):
        self.eval()
        # encoding the source code
        BLOCK_SIZE = 400
        code = ' '.join(code.split())
        code_tokens = []
        source_tokens = []
        if self.args.model_type == 'codegptsensor':
            code_tokens = self.tokenizer.tokenize(code)[:BLOCK_SIZE-4]
            source_tokens = [self.tokenizer.cls_token,"<encoder_only>",self.tokenizer.sep_token] + code_tokens + [self.tokenizer.sep_token]
        elif self.args.model_type == 'codebert':
            code_tokens = self.tokenizer.tokenize(code)[:BLOCK_SIZE-2]
            source_tokens = [self.tokenizer.cls_token]+code_tokens+[self.tokenizer.sep_token]

        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = BLOCK_SIZE - len(source_ids)
        source_ids += [self.tokenizer.pad_token_id] * padding_length

        # predict the label
        source_ids = torch.Tensor([source_ids]).long().to("cuda")
        prob = self.forward(source_ids)[0]
        prob = prob.cpu().detach().numpy()
        return prob