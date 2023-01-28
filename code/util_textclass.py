import os
import torch
import logging
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class Instance:

    def __init__(self, text, label):
        self.text = text
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

def read_instances_from_file(args, data_dir, mode, delimiter="\t"):
    file_path = os.path.join(data_dir, "{}.tsv".format(mode))
    instances = []

    line_data  = pd.read_csv(file_path, sep=delimiter)

    texts = line_data['text'].values
    labels = line_data['category'].values
    headlines  = line_data['headline'].values

    for text_, headline_, label_ in zip(texts, headlines, labels):
        if int(args.header) == 1:
            text_ = headline_.strip() + ". " + text_.strip()
        instances.append(Instance(text_, label_))

    return instances

def convert_instances_to_features_and_labels(instances, tokenizer, labels, max_seq_length):
    label_map = {label: i for i, label in enumerate(labels)}

    features = []
    for instance_idx, instance in enumerate(instances):
        tokenization_result = tokenizer.encode_plus(text=instance.text,
                                                    max_length=max_seq_length, pad_to_max_length=True,
                                                     truncation=True)
        token_ids = tokenization_result["input_ids"]
        try:
            token_type_ids = tokenization_result["token_type_ids"]
        except:
            token_type_ids = None
        attention_masks = tokenization_result["attention_mask"]

        if instance.label not in label_map:
            continue
        label = label_map[instance.label]

        if "num_truncated_tokens" in tokenization_result:
            logger.info(f"Removed {tokenization_result['num_truncated_tokens']} tokens from {instance.text} as they "
                         f"were longer than max_seq_length {max_seq_length}.")

        if instance_idx < 3:
            logger.info("Tokenization example")
            logger.info(f"  text: {instance.text}")
            logger.info(f"  tokens (by input): {tokenizer.tokenize(instance.text)}")
            logger.info(f"  token_ids: {tokenization_result['input_ids']}")
            #logger.info(f"  token_type_ids: {tokenization_result['token_type_ids']}")
            logger.info(f"  attention mask: {tokenization_result['attention_mask']}")

        features.append(
            InputFeatures(input_ids=token_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, label=label)
        )

    return features

def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        return labels
    else:
        return ['sports', 'health', 'technology', 'business', 'politics', 'entertainment', 'religion', 'uncategorized']