import torch
import numpy as np
import math

from .utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from ..preprocessing.process_data import preprocess_data
import os

class FewShotModelTester():
    def __init__(
        self, 
        name, 
        model='light_state.pt',
        model_path=DEFAULT_MODEL_PATH,
        path=DEFAULT_DATASET_SAVING_PATH,
        output_type='Default', 
        debug=False
    ):
        self.model_loc = model_path + model
        self.location = path + name.replace('/', '-')
        self.output_type = output_type
        self.debug = debug

        self.set_up_model()

    def process_train_set(self):
        self.train = torch.load(self.location + '/train_set.pt')

        _, prompt_components, outputs = self.train
        prompts, annotations, labels = prompt_components
        self.gt = _[1]
        samples = [{
            'prompt': prompts[i],
            'annotations': annotations[i],
            'labels': labels[i],
            'output': outputs[i],
        } for i in range(len(prompts))]

        self.train = samples

    def process_test_set(self):
        self.test = torch.load(self.location + '/test_set.pt')

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

    
    def fine_tune(self, epochs=30, dataset_name='default'):
        self.process_train_set()

        self.model.train()
        data_module = make_supervised_data_module(tokenizer=self.tokenizer, data=self.train, prompt_type=self.output_type)
        training_args = TrainingArguments("files/model_checkpoints", num_train_epochs=epochs)
        training_args = training_args.set_save(strategy="steps", steps=10000, total_limit=10)


        trainer = Trainer(model=self.model, tokenizer=self.tokenizer, args=training_args, **data_module)
        trainer.train()

        custom_weights_path = DEFAULT_MODEL_PATH + 'light_state_'+ dataset_name + ".pt"
        torch.save(self.model.state_dict(), custom_weights_path)

    
    def make_prompt(self):     
        if self.output_type == 'TabLLM':
            for item in self.test:
                item['labels'] = item['labels'][item['labels'].index('where'): ]

        prompt_input = PROMPT_DICT[self.output_type]
        self.prompts = [prompt_input.format_map(example) for example in self.test]
        # print(self.prompts[0])

    def extract_probabilities(self, reference, pred):
        ref_match = re.search(r'class 1: ([0-9.]+)', reference)
        pred_match = re.search(r'class 1: ([0-9.]+)', pred)

        ref_prob_class1 = float(ref_match.group(1).strip('.')) if ref_match else 0
        pred_prob_class1 = float(pred_match.group(1).strip('.')) if pred_match else 0

        true_label = 1 if ref_prob_class1 >= 0.5 else 0
        return true_label, pred_prob_class1

    def get_model_accuracy(self):
        self.process_test_set()

        tblm = True if self.output_type=='TabLLM' else False
        self.model.eval()
        correct_preds = 0
        batch_y_true = []
        batch_y_pred = []
        for i in range(len(self.prompts)):
            prompt = self.prompts[i]
            reference = self.test[i]['output']
            pred = self.test_model_on_one_prompt(prompt).split('\n')[-1]
            true_label, pred_prob = self.extract_probabilities(reference, pred)
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
        fpr, tpr, thresholds = roc_curve(batch_y_true, batch_y_pred)
        auc_score = auc(fpr, tpr)
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
    
    def set_up_model(self):
        self.model, self.tokenizer = setup_model_and_tokenizer('gpt2')
        self.model.load_state_dict(torch.load(self.model_loc, map_location='cuda:0'))

class FewShotTester():
    def __init__(
        self, 
        model='unipred_state.pt',
        dataset_list='few_shot_datasets.json',
        model_path=DEFAULT_MODEL_PATH,
        dataset_list_path = DEFAULT_DATASET_INDEXING_PATH,
        dataset_path=DEFAULT_DATASET_SAVING_PATH, 
        output_type='Default', 
        debug=False,
        test_mode=False,
    ):
        self.model = model
        self.model_path = model_path
        self.dataset_list_loc = dataset_list_path + dataset_list
        self.dataset_path = dataset_path
        self.dataset_list = read_json(self.dataset_list_loc)
        if test_mode:
            self.dataset_list = self.dataset_list[:3]
        self.output_type = output_type
        self.debug = debug
        self.acc_dict = {}
        self.delta_acc = 0
    
    def fine_tune(self, dataset_name):
        try:
            tester = FewShotModelTester(
                            dataset_name, 
                            model=self.model, 
                            model_path=self.model_path, 
                            path=self.dataset_path, 
                            output_type=self.output_type, 
                            debug=self.debug
                        )
            tester.fine_tune(dataset_name=dataset_name)
        except Exception as e:
            if self.debug:
                print('=======debug output=======')
                print(e)
    
    def get_accuracy(self):
        for item in self.dataset_list:
            try:
                tester = FewShotModelTester(
                    item, 
                    model=self.model, 
                    model_path=self.model_path, 
                    path=self.dataset_path, 
                    output_type=self.output_type, 
                    debug=self.debug
                )
                tester.get_model_accuracy()
                acc = tester.accuracy
                auc = tester.auc_score
                if self.debug:
                    print(acc)
                return auc,acc
            except Exception as e:
                if self.debug:
                    print('=======debug output=======')
                    print(e)
                continue

    def load_acc_dict(self, path='files/unified/results/few_shot.json'):
        try:
            self.acc_dict = read_json(path)
        except:
            self.acc_dict = {}
    
    def save_acc_dict(self, path='files/unified/results/few_shot.json'):
        save_json(path, self.acc_dict)


class Light():
    def __init__(self,dataset_json='dataset.json'):
        self.dataset_json = dataset_json

    def train(self, dataset_name, train_set):
        file_path = DEFAULT_DATASET_INDEXING_PATH + self.dataset_json
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        dataset_content = [dataset_name]
        with open(file_path, 'w') as json_file:
            json.dump(dataset_content, json_file, indent=4)

        preprocess_data(data=train_set,dataset_name=dataset_name,set_type='train')

        model_weight_path = 'light_state.pt'
        st = FewShotTester(dataset_list= self.dataset_json , model=model_weight_path, output_type='light', debug=True)
        st.fine_tune(dataset_name)

    def test(self, dataset_name, test_set):
        file_path = DEFAULT_DATASET_INDEXING_PATH +self.dataset_json
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        dataset_content = [dataset_name]
        with open(file_path, 'w') as json_file:
            json.dump(dataset_content, json_file, indent=4)

        preprocess_data(data=test_set, dataset_name=dataset_name, set_type='test')

        model_weight_path = 'light_state_' + dataset_name + '.pt'
        model_weight_path = 'light_state1.pt'
        st = FewShotTester(dataset_list= self.dataset_json , model=model_weight_path, output_type='light', debug=False)
        auc,acc = st.get_accuracy()

        return auc,acc