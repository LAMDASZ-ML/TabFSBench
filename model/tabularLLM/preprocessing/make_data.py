import torch
import json
from transformers import Trainer, GPT2LMHeadModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from .dataset import *
from .preprocess_kaggle_dataset import *

DEFAULT_PROMPT_SAVING_PATH = 'files/unified/prompts/'

class Prompt_maker(DataObject):
    def __init__(self, name, output_type='Default', path=None, from_preprocessed=True, debug=False, max_len=7500, min_len=500, zero_shot=100):
        super().__init__(name, path, from_preprocessed, output_type=output_type)
        self.prompt = None
        self.prompt_cat = None
        self.debug = debug
        self.max_len = max_len
        self.min_len = min_len
        self.zero_shot = zero_shot

    def make_prompt(self):
        _, prompt_components, outputs = self.train
        prompts, annotations, labels = prompt_components
        if len(prompts[0]) + len(annotations[0]) + len(labels[0]) > 4000:
            if self.debug:
                print(f'Dataset {self.name} too long: {len(prompts[0]) + len(annotations[0]) + len(labels[0])} chars total.')
            return

        samples = [{
            'prompt': prompts[i],
            'annotations': annotations[i],
            'labels': labels[i],
            'output': outputs[i],
        } for i in range(len(prompts))]

        if len(samples) > self.max_len:
            if self.debug:
                print('Sample size too large. Resampling...')
            samples = resample(samples, n_samples=self.max_len, replace=False, random_state=42)
        elif len(samples) < self.zero_shot:
            if self.debug:
                print('Zero-shot dataset created.')
            self.prompt_cat = 'zero-shot'
            self.prompt = samples
            return
        elif len(samples) > self.min_len:
            if self.debug:
                print('Supervised dataset created.')
            self.prompt_cat = 'supervised'
            self.prompt = samples
        if self.debug:
            print('Sample size < supervised dataset requirement but > zero shot dataset requirement. Omitting...')
    
    def get_prompt(self):
        return self.prompt
    
    def get_state(self):
        return self.prompt_cat
    

class Prompt_generator():
    def __init__(self, metadata_path=None, saving_path=None, debug=False, dataset_info_list_path='files/unified/dataset_list/datasets_after_round_2.json'):
        self.debug = debug
        self.save_path = metadata_path if metadata_path else DEFAULT_PROMPT_SAVING_PATH
        self.dataset_path = saving_path if saving_path else DEFAULT_DATASET_SAVING_PATH
        if dataset_info_list_path:
            self.dataset_info_list = read_json(dataset_info_list_path)
        else:
            self.dataset_info_list = None
        self.prompt_makers = []
        self.zero_shot_prompts = []
        self.supervised_prompts = []
        self.few_shot_datalist = []
        self.supervised_datalist = []

    def make_all_prompts(self, output_type='Default'):
        self.prompt_makers = []
        for item in self.dataset_info_list:
            try:
                pm = Prompt_maker(item, debug=self.debug, output_type=output_type)
                pm.make_prompt()
                self.prompt_makers.append(pm)
            except Exception as e:
                if self.debug:
                    print(f'prompt maker for {item} stopped working due to {e}.')
                continue
            if pm.get_state() == 'zero-shot':
                self.zero_shot_prompts.extend(pm.get_prompt())
                self.few_shot_datalist.append(item)
            if pm.get_state() == 'supervised':
                self.supervised_prompts.extend(pm.get_prompt())
                self.supervised_datalist.append(item)

    def tokenize_supervised_prompt(self, prompt_type='prompt_input'):
        model, tokenizer = setup_model_and_tokenizer('gpt2')
        self.supervised_data_module = make_supervised_data_module(tokenizer=tokenizer, data=self.supervised_prompts, prompt_type=prompt_type)
    
    def save_tokenized_prompt(self, name='toked_train_set.pt'):
        torch.save(self.supervised_data_module, self.save_path + name)
        save_json('files/unified/dataset_list/few_shot_datasets.json', self.few_shot_datalist)
        save_json('files/unified/dataset_list/supervised_datasets.json', self.supervised_datalist)
        
    def show_prompt(self):
        print(self.supervised_prompts[0])

def make_full_dataset(max_len=7500, min_len=500, zero_shot=100):
    with open('files/data/kaggle_dataset_record.json', 'r') as f:
        metadata_list = json.load(f)
    dataset_list = []
    train_set = []
    zero_shot_test = []
    # test_set = []

    for item in metadata_list:
        author, path = item.split('/')
        path = f'{author}-{path}'
        print(f'Loading {path}...')
        try:
            _, prompt_components, outputs = torch.load(f'files/data/kaggle/{path}/train_set.pt')
            prompts, annotations, labels = prompt_components
        except Exception as e:
            print(f'Failed due to {e}. Skipping...')
            continue
        #[{'prompt': train_prompts[i], 'output': train_outputs[i], 'annotations': train_annotations[i], 'labels': train_labels[i]} for i in range(len(train_prompts))]
        
        if len(prompts[0]) + len(annotations[0]) + len(labels[0]) > 4000:
            print(f'Dataset {item} too long: {len(prompts[0]) + len(annotations[0]) + len(labels[0])} chars total.')
            continue

        samples = [{
            'prompt': prompts[i],
            # 'annotations': annotations[i],
            'annotations': 'None',
            'labels': labels[i],
            'output': outputs[i],
        } for i in range(len(prompts))]

        if len(samples) > max_len:
            samples = resample(samples, n_samples=7500, replace=False, random_state=42)
        elif len(samples) < zero_shot:
            zero_shot_test.extend(samples)
            continue
        elif len(samples) > min_len:
            train_set.extend(samples)
            dataset_list.append((path, len(samples)))
            print(len(train_set))

    # os.mkdir('files/data/processed/trial_1')
    # with open('files/data/processed/trial_1/dataset_info.json', 'w+') as f:
    #     json.dump(dataset_list, f, indent=4)
    # with open('files/data/processed/trial_1/untok_train_kaggle.json', 'w+') as f:
    #     json.dump(train_set, f, indent=4)
    # with open('files/data/processed/trial_1/zero_shot_test.json', 'w+') as f:
    #     json.dump(zero_shot_test, f, indent=4)
    model, tokenizer = setup_model_and_tokenizer('gpt2')
    data_module = make_supervised_data_module(tokenizer=tokenizer, data=train_set)
    torch.save(data_module, 'files/data/processed/trial_1/ablation.pt')
            
if __name__ == '__main__':
    make_full_dataset()