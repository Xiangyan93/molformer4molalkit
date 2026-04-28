#!/usr/bin/env python
# -*- coding: utf-8 -*-
from transformers import PreTrainedTokenizer
import regex as re


PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"


class MolTranBertTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file: str = '',
                 do_lower_case=False,
                 unk_token='<pad>',
                 sep_token='<eos>',
                 pad_token='<pad>',
                 cls_token='<bos>',
                 mask_token='<mask>',
                 **kwargs):
        self._vocab = {}
        self._ids_to_tokens = {}
        with open(vocab_file, 'r') as f:
            for idx, line in enumerate(f):
                token = line.strip()
                self._vocab[token] = idx
                self._ids_to_tokens[idx] = token

        self.regex_tokenizer = re.compile(PATTERN)

        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

    @property
    def vocab(self):
        return dict(self._vocab)

    @property
    def vocab_size(self):
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def _tokenize(self, text):
        return self.regex_tokenizer.findall(text)

    def _convert_token_to_id(self, token):
        return self._vocab.get(token, self._vocab.get(self.unk_token, 0))

    def _convert_id_to_token(self, index):
        return self._ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens).strip()
