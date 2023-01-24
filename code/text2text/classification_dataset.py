
from typing import Dict
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ClassificationDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, data_path:str,  **kwargs):
        
        self.tokenizer = tokenizer
        self.__dict__.update(kwargs)
        if data_path.endswith(".tsv"):
            self.data = pd.read_csv(data_path, sep='\t')
            if 'ID' not in self.data.columns:
                self.data['ID'] = range(1, len(self.data) + 1)
        elif data_path.endswith(".csv"):
            self.data = pd.read_csv(data_path)
            if 'ID' not in self.data.columns:
                self.data['ID'] = range(1, len(self.data) + 1)

        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)
    
    def get_count(self):
        return self.data.groupby(self.target_column)['ID'].nunique().values.tolist()

    def __getitem__(self, index: int):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self) -> None:
        print("[INFO] Building dataset...")
        for idx in range(len(self.data)):
            input_, target = self.data.loc[idx,self.data_column], self.data.loc[idx, self.target_column]

            target = self.class_map[target.lower()]
            
            input_ = self.prompt + input_.lower() + ' </s>'
            target = target + " </s>"

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=self.max_seq_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=2, padding="max_length", truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)
        print("[INFO] Done Building the Dataset")
        
        
        

        
        
class ClassificationDatasetTest(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, data_path:str,  **kwargs):
        
        self.tokenizer = tokenizer
        self.__dict__.update(kwargs)
        if data_path.endswith(".tsv"):
            self.data = pd.read_csv(data_path, sep='\t')
            if 'ID' not in self.data.columns:
                self.data['ID'] = range(1, len(self.data) + 1)
        elif data_path.endswith(".csv"):
            self.data = pd.read_csv(data_path)  
            if 'ID' not in self.data.columns:
                self.data['ID'] = range(1, len(self.data) + 1)

        self.inputs = []
        self.id = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        # target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()
        
        column_id = self.id[index]
        # target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "column_id": column_id } #, "target_mask": target_mask}

    def _build(self) -> None:
        print("[INFO] Building dataset...")
        for idx in range(len(self.data)):
            input_,column_id = self.data.loc[idx,self.data_column] , self.data.loc[idx,'ID']

            # target = self.class_map[target.lower()]
            
            input_ = self.prompt + input_.lower() + ' </s>'
            # target = target + " </s>"

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=self.max_seq_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            # tokenize targets
            # tokenized_targets = self.tokenizer.batch_encode_plus(
            #     [target], max_length=2, padding="max_length", truncation=True, return_tensors="pt"
            # )

            self.inputs.append(tokenized_inputs)
            self.id.append(column_id)
            # self.targets.append(tokenized_targets)
        print("[INFO] Done Building the Dataset")
