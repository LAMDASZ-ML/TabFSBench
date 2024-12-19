import torch
import pandas as pd
import json
import copy
import os
import sys
import transformers
import logging

from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from torch.utils.data import Dataset
from transformers import Trainer, GPT2LMHeadModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments
from sklearn.model_selection import train_test_split


# GLOBAL VARIABLES
PROMPT_DICT = {
    "prompt_input": (
        "Below is the description of a dataset, an object profile from the dataset and a target description. "
        "Predict the target by the given information of the object.\n"
        "# Dataset description: {annotations}\n"
        "# Object description: {prompt}\n"
        "# You should return the probability of each class by: \n{labels}\n"
        "# Answer: \n"
    ),
    "without_metadata": (
        "Below is a dataset. Predict the target by the given information of the object.\n"
        "# Object description: {prompt}\n"
        "# You should return the probability of each class by: \n{labels}\n"
        "# Answer: \n"
    ),
    "TabLLM": (
        "Below is the description of a dataset, an object profile from the dataset and a target description. "
        "Predict the target by the given information of the object.\n"
        "# Dataset description: {annotations}\n"
        "# Object description: {prompt}\n"
        "# You should return your choice of class by stating the class number, {labels}\n"
        "# Answer: \n"
    )
}
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_MASK_TOKEN = "[MASK]"


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """
    Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """
    Tokenize a list of strings.
    """
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Preprocess the data by tokenizing them.
    """
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    # TODO: Change to IGNORE_INDEX
    # TODO: Reverse label[:]
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning.
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data, prompt_type):
        super(SupervisedDataset, self).__init__()
        list_data_dict = data
        logging.warning("Formatting inputs...")
        if prompt_type == 'TabLLM':
            for item in list_data_dict:
                item['labels'] = item['labels'][item['labels'].index('where'): ]

        prompt_input = PROMPT_DICT[prompt_type]
        sources = [prompt_input.format_map(example) for example in list_data_dict]
        print(sources[0])
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """
    Collate examples for supervised fine-tuning.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data, prompt_type='prompt_input') -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    """
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data=data, prompt_type=prompt_type)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def setup_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

    tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=1024,
            padding_side="left",
            use_fast=False,
        )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN, mask_token=DEFAULT_MASK_TOKEN),
            tokenizer=tokenizer,
            model=model,
    )
    return model, tokenizer

