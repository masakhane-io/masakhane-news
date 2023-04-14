import os
from tqdm import tqdm
from typing import List
import pandas as pd
import json
import csv


def load_train_data(language: str):
    assert language in get_available_languages(), 'Requested language is not available'
    df = pd.read_csv(f'../data/{language}/train.tsv',
                     sep='\t', usecols=['category', 'headline', 'text'])
    return df


def load_validation_data(language: str):
    assert language in get_available_languages(), 'Requested language is not available'
    df = pd.read_csv(f'../data/{language}/dev.tsv', sep='\t',
                     usecols=['category', 'headline', 'text'])
    return df


def load_test_data(language: str):
    assert language in get_available_languages(), 'Requested language is not available'
    df = pd.read_csv(f'../data/{language}/test.tsv', sep='\t',
                     usecols=['category', 'headline', 'text'])
    return df


def get_available_languages():
    return list(os.listdir('../data'))


def get_labels(language: str):
    with open(f'../data/{language}/labels.txt', 'r') as f:
        labels = f.readlines()
    return [label.replace('\n', '') for label in labels]


def get_samples_per_class(df: pd.DataFrame, n: int):
    return df.groupby('category', as_index=False).apply(lambda cat: cat.sample(n, random_state=42)).reset_index(drop=True)

if __name__ == '__main__':
    if os.path.exists('../few_shot_data'):
        raise Exception('Remove previous experiment run data')

    os.mkdir('../few_shot_data')
    os.mkdir('../few_shot_data/5_samples')
    os.mkdir('../few_shot_data/10_samples')
    os.mkdir('../few_shot_data/20_samples')
    os.mkdir('../few_shot_data/50_samples')

    for language in tqdm(get_available_languages()):
        print(f'Processing language {language}')
        for num_samples in [5, 10, 20, 50]:
            train_data = load_train_data(language)
            test_data = load_test_data(language)
            dev_data = load_validation_data(language)
            path = f'../few_shot_data/{num_samples}_samples/{language}'
            os.mkdir(path)
            test_data_local = test_data
            train_data_local = get_samples_per_class(train_data, num_samples)
            dev_data_local = get_samples_per_class(dev_data, 5)
            dev_data_local.to_csv(path+'/dev.tsv', sep='\t', index=False)
            train_data_local.to_csv(path+'/train.tsv', sep='\t', index=False)
            test_data_local.to_csv(path+'/test.tsv', sep='\t', index=False)
            os.popen(
                f'cp ../data/{language}/labels.txt ../few_shot_data/{num_samples}_samples/{language}'
            )
