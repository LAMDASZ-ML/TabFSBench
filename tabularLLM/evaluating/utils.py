import torch
import json
import numpy as np
import re
import transformers
import copy

from dataclasses import dataclass, field
from pandas.api.types import is_numeric_dtype, is_integer_dtype
from typing import Optional, Dict, Sequence
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
from transformers import Trainer, GPT2LMHeadModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments

PROMPT_DICT = {
    "Default": (
        "Below is the description of a dataset, an object profile from the dataset and a target description. "
        "Predict the target by the given information of the object.\n"
        "# Dataset description: {annotations}\n"
        "# Object description: {prompt}\n"
        "# You should return the probability of each class by: \n{labels}\n"
        "# Answer: \n"
    ),
    "Ablation_aug": (
        "Below is a dataset. Predict the target by the given information of the object.\n"
        "# Object description: {prompt}\n"
        "# You should return the probability of each class by: \n{labels}\n"
        "# Answer: \n"
    ),
    "light": (
        "Below is a dataset. Predict the target by the given information of the object.\n"
        "# Object description: {prompt}\n"
        "# You should return the probability of each class by: \n{labels}\n"
        "# Answer: \n"
    ),
    "TabLLM": (
        "Below is a dataset. Predict the target by the given information of the object.\n"
        "# Object description: {prompt}\n"
        "# You should return your choice of class by stating the class number, {labels}\n"
        "# Answer: \n"
    )
}
DEFAULT_DATASET_INDEXING_PATH = 'files/unified/dataset_list/'
DEFAULT_MODEL_PATH = 'files/unified/models/'
DEFAULT_DATASET_SAVING_PATH = 'dataset/'
DEFAULT_PROMPT_SAVING_PATH = 'files/unified/prompts/'
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_MASK_TOKEN = "[MASK]"

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_json(path, data):
    with open(path, 'w+') as f:
        json.dump(data, f, indent=4)


def data_to_prompt(data, column_names):
    listed_data = data.values.tolist()
    prompts = []
    for row in listed_data:
        prompt = []
        for i in range(len(column_names)):
            column = column_names[i].replace('_', ' ')
            value = row[i]
            prompt_segment = f'{column} is {value}; '
            prompt.append(prompt_segment)
        prompt = ''.join(prompt)[:-2] + '.\n'
        prompts.append(prompt)
    return prompts


def label_to_prompt(label, length):
    classification = []
    prompt = []
    for key in label.keys():
        classification.append(f'class {label[key]}: xxx; ')
        prompt_segment = f'class {label[key]} stands for "{key}"; '
        prompt.append(prompt_segment)
    # Class 0 is xxx; class 1 is xxx; where class 0 stands for yyy; class 1 stands for yyy.
    full_prompt = ''.join(classification) + 'where ' + ''.join(prompt)[:-2] + '.\n'
    return [full_prompt] * length


def numericalize(samples, labels, column_names):
    samples = samples.reset_index(drop=True)
    for i in range(len(column_names)):
        col = column_names[i]
        if is_numeric_dtype(samples[col]) or is_integer_dtype(samples[col]):
            continue
        categories = [cat for cat in set(samples[col].to_list())]
        cat_dict = {categories[i]: i for i in range(len(categories))}
        samples[col] = samples[col].map(cat_dict).astype(int)
    categories = [cat for cat in set(labels.to_list())]
    cat_dict = {categories[i]: i for i in range(len(categories))}
    labels = labels.map(cat_dict).astype(int)
    return samples.to_numpy(), labels.to_numpy().reshape(-1, 1)


def calculate_auc(labels, preds):
    # ground truth
    if len(labels.shape) > 1:
        y_gt = labels.squeeze(1)
    else: 
        y_gt = labels
    onehot = np.zeros((y_gt.size, y_gt.max().astype(int) + 1))
    onehot[np.arange(y_gt.size), y_gt] = 1
    y_gt = onehot
    # pred
    y_pred = preds
    return roc_auc_score(y_gt, y_pred, average=None)

def serialize_output(preds):
    outputs = []
    for i in preds:
        out_strs = []
        for j, k in enumerate(i):
            out_str = f'class {j}: {np.round(k, 2)}; '
            out_strs.append(out_str)
        out_strs = ''.join(out_strs)
        out_strs = out_strs[:-2] + '.'
        outputs.append(out_strs)
    return outputs

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
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning.
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data, prompt_type):
        super(SupervisedDataset, self).__init__()
        list_data_dict = data
        if prompt_type == 'TabLLM':
            for item in list_data_dict:
                item['labels'] = item['labels'][item['labels'].index('where'): ]

        prompt_input = PROMPT_DICT[prompt_type]
        sources = [prompt_input.format_map(example) for example in list_data_dict]
        # print(sources[0])
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

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


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data, prompt_type='Default') -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    """
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data=data, prompt_type=prompt_type)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
    # try:
    #     # 创建训练数据集和数据处理器
    #     train_dataset = SupervisedDataset(tokenizer=tokenizer, data=data, prompt_type=prompt_type)
    #     data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    #     # 打印类型以确认
    #     print(f"train_dataset type: {type(train_dataset)}")
    #     print(f"data_collator type: {type(data_collator)}")

    #     # 确保返回的是字典而不是元组
    #     result = {
    #         'train_dataset': train_dataset, 
    #         'eval_dataset': None,  # 假设没有评估数据集
    #         'data_collator': data_collator
    #     }

    #     # 打印返回的字典类型
    #     print(f"Result type: {type(result)}")

    #     return result

    # except Exception as e:
    #     # 捕获异常并打印
    #     print(f"Error occurred: {e}")
    #     return {}


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

def response_to_class(response):
    return re.findall(r'[\d]*[.][\d]+', response)

def check_correctness(pred, ref, tblm='False'):
    if tblm:
        pred_cls = re.findall(r'[\d]+', pred)
        ref_cls = re.findall(r'[\d]+', ref)
        if pred_cls == ref_cls:
            return True
        else:
            return False
    else:
        try:
            pred_cls = response_to_class(pred)
            ref_cls = response_to_class(ref)
            # print(pred_cls)
            # print(ref_cls)
            pred_idx = pred_cls.index(max(pred_cls))
            ref_idx = ref_cls.index(max(ref_cls))
            # print(pred_idx)
            # print(ref_idx)
            if pred_idx == ref_idx:
                return True
        except:
            return False
        return False

