import torch
import numpy as np
import math

from .utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from preprocessing.process_data import preprocess_data
import os
import tqdm

class TabLLMTester():
    def __init__(
        self, 
        name, 
        path=DEFAULT_DATASET_SAVING_PATH,
        output_type='TabLLM', 
        model_weight = 'default',
        debug=False
    ):
        self.location = path + name.replace('/', '-')
        self.output_type = output_type
        self.debug = debug

        self.set_up_model(model_weight)

    def reshape_output(self):
        output = [f'class {int(item)}' for item in self.train[0][1]]
        self.train = (self.train[0], self.train[1], output)
        output = [f'class {int(item)}' for item in self.test[0][1]]
        self.test = (self.test[0], self.test[1], output)
        
    def merge_data(self):
        _, prompt_components, outputs = self.train
        prompts, annotations, labels = prompt_components

        samples = [{
            'prompt': prompts[i],
            'annotations': annotations[i],
            'labels': labels[i],
            'output': outputs[i],
        } for i in range(len(prompts))]

        _, prompt_components, outputs = self.test
        prompts, annotations, labels = prompt_components

        samples2 = [{
            'prompt': prompts[i],
            'annotations': annotations[i],
            'labels': labels[i],
            'output': outputs[i],
        } for i in range(len(prompts))]

        samples.extend(samples2)
        self.samples = samples

    def split(self, train_ratio):
        self.train, self.test = train_test_split(
            self.samples,
            train_size=train_ratio,
            random_state=42
        )

    def process_train_set(self):
        self.train = torch.load(self.location + '/train_set.pt')
        output = [f'class {int(item)}' for item in self.train[0][1]]
        self.train = (self.train[0], self.train[1], output)

        _, prompt_components, outputs = self.train
        prompts, annotations, labels = prompt_components

        samples = [{
            'prompt': prompts[i],
            'annotations': annotations[i],
            'labels': labels[i],
            'output': outputs[i],
        } for i in range(len(prompts))]
        self.train = samples

    def process_test_set(self):
        self.test = torch.load(self.location + '/test_set.pt')
        output = [f'class {int(item)}' for item in self.test[0][1]]
        self.test = (self.test[0], self.test[1], output)

        _, prompt_components, outputs = self.test
        prompts, annotations, labels = prompt_components

        samples2 = [{
            'prompt': prompts[i],
            'annotations': annotations[i],
            'labels': labels[i],
            'output': outputs[i],
        } for i in range(len(prompts))]
        self.test = samples2
        self.make_prompt()    
        

    def fine_tune(self, epochs=3, dataset='default'):
        self.process_train_set()

        data_module = make_supervised_data_module(tokenizer=self.tokenizer, data=self.train, prompt_type='TabLLM')
        training_args = TrainingArguments("files/checkpoints", num_train_epochs=epochs)
        training_args = training_args.set_save(strategy="steps", steps=10000, total_limit=10)

        trainer = Trainer(model=self.model, tokenizer=self.tokenizer, args=training_args, **data_module)
        trainer.train()

        custom_weights_path = DEFAULT_MODEL_PATH + 'tabllm_state_'+dataset+".pt"
        torch.save(self.model.state_dict(), custom_weights_path)
        print("model saved!")
    
    def make_prompt(self):     
        for item in self.test:
            item['labels'] = item['labels'][item['labels'].index('where'): ]

        prompt_input = PROMPT_DICT[self.output_type]
        self.prompts = [prompt_input.format_map(example) for example in self.test]
    

    def get_model_accuracy(self):
        self.process_test_set()

        tblm = True
        correct_preds = 0
        batch_y_true = []
        batch_y_pred = []
        for i in range(len(self.prompts)):
            prompt = self.prompts[i]
            reference = self.test[i]['output']
            pred = self.test_model_on_one_prompt(prompt).split('\n')[-1]
            true_label = 1 if reference == 'class 1' else 0
            pred_prob = 1 if pred == 'class 1' else 0
            batch_y_true.append(true_label)
            batch_y_pred.append(pred_prob)
            if self.debug:
                print(pred)
                print(reference)
            corr = check_correctness(pred, reference, tblm=tblm)
            if self.debug:
                print(corr)
            correct_preds += corr
        if self.debug:
            print(correct_preds / len(self.prompts))
        self.accuracy = correct_preds / len(self.prompts)
        auc_score = roc_auc_score(batch_y_true, batch_y_pred)
        self.auc_score = auc_score
    
    def test_model_on_one_prompt(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        # print(inputs, inputs['input_ids'].squeeze(0).shape, tokenizer.decode(inputs['input_ids'].squeeze(0)))
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            max_length=1024,
            top_k=50,
            top_p=0.95,
            num_return_sequences=3,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def clear_model(self):
        self.tokenizer = None
        self.model = None
    
    def set_up_model(self, model_weight):
        self.model, self.tokenizer = setup_model_and_tokenizer('gpt2')

        if model_weight != 'default':
            if os.path.exists(model_weight):
                self.model.load_state_dict(torch.load(model_weight, map_location='cuda:0'))
                print("model load success!")



class TabLLMTesterGroup():
    def __init__(
        self, 
        dataset_list='few_shot_datasets.json',
        dataset_list_path = DEFAULT_DATASET_INDEXING_PATH,
        dataset_path=DEFAULT_DATASET_SAVING_PATH, 
        output_type='TabLLM', 
        debug=False,
        test_mode=False,
        model_weight = 'default'
    ):
        self.dataset_list_loc = dataset_list_path + dataset_list
        self.dataset_path = dataset_path
        self.dataset_list = read_json(self.dataset_list_loc)
        if test_mode:
            self.dataset_list = self.dataset_list[:3]
        self.output_type = output_type
        self.debug = debug
        self.acc_dict = {}
        self.model_weight = model_weight
    
    def fine_tune(self, dataset):
        try:
            tester = TabLLMTester(
                    dataset, 
                    path=self.dataset_path, 
                    output_type=self.output_type, 
                    debug=self.debug
                )
            tester.fine_tune(epochs=30,dataset=dataset)
        except Exception as e:
            if self.debug:
                print('=======debug output=======')
                print(e)

    def get_supervised_accuracy(self):
        for item in self.dataset_list:
            try:
                tester = TabLLMTester(
                    item,
                    path=self.dataset_path, 
                    output_type=self.output_type, 
                    debug=self.debug,
                    model_weight=self.model_weight
                )
                tester.fine_tune()
                tester.get_model_accuracy()
                acc = tester.accuracy
                if self.debug:
                    print(acc)
                acc = tester.accuracy
                if item in self.acc_dict.keys():
                    self.acc_dict[item][self.output_type] = acc
                else:
                    self.acc_dict[item] = {self.output_type: acc}
            except Exception as e:
                if self.debug:
                    print('=======debug output=======')
                    print(e)
                continue
    
    def get_few_shot_accuracy(self):
        for item in self.dataset_list:
            try:
                tester = TabLLMTester(
                    item, 
                    path=self.dataset_path, 
                    output_type=self.output_type, 
                    debug=self.debug,
                    model_weight=self.model_weight
                )
                tester.get_model_accuracy()
                acc = tester.accuracy
                auc = tester.auc_score
                if self.debug:
                    print(f"accuracy:{acc}")
                    print(f"auc_score:{auc}")
                return auc,acc
            except Exception as e:
                if self.debug:
                    print('=======debug output=======')
                    print(e)
                continue

    def load_acc_dict(self, path='files/unified/results/tblm.json'):
        try:
            self.acc_dict = read_json(path)
        except:
            self.acc_dict = {}
    
    def save_acc_dict(self, path='files/unified/results/tblm.json'):
        save_json(path, self.acc_dict)


    

class TabLLM():
    def __init__(self,dataset_json='dataset.json'):
        self.dataset_json = dataset_json

    def train(self, dataset_name,train_set):
        file_path = DEFAULT_DATASET_INDEXING_PATH + self.dataset_json
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        dataset_content = [dataset_name]
        with open(file_path, 'w') as json_file:
            json.dump(dataset_content, json_file, indent=4)

        preprocess_data(data=train_set,dataset_name=dataset_name,set_type='train')

        tg = TabLLMTesterGroup(dataset_list=self.dataset_json, debug=True)
        tg.fine_tune(dataset_name)

    def test(self, dataset_name, test_set):
        file_path = DEFAULT_DATASET_INDEXING_PATH +self.dataset_json
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        dataset_content = [dataset_name]
        with open(file_path, 'w') as json_file:
            json.dump(dataset_content, json_file, indent=4)

        preprocess_data(data=test_set, dataset_name=dataset_name, set_type='test')

        model_weight_path = DEFAULT_MODEL_PATH + 'tabllm_state_' + dataset_name + '.pt'
        tg = TabLLMTesterGroup(dataset_list=self.dataset_json, debug=False, model_weight=model_weight_path)
        auc,acc =tg.get_few_shot_accuracy()

        return auc,acc

        

        

            