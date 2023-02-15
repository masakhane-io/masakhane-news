from datasets import load_dataset, Features, Value, ClassLabel
from setfit import SetFitModel, SetFitTrainer, sample_dataset
from sklearn.metrics import f1_score
import os
from typing import List
from tqdm import tqdm
import pandas as pd
import json
import csv
import evaluate
import os
from datasets import Dataset, Features, Value, ClassLabel
from sentence_transformers.losses import CosineSimilarityLoss


def load_train_data(language: str, use_article_text: bool):
    assert language in get_available_languages(), 'Requested language is not available'
    df = pd.read_csv(f'../../data/{language}/train.tsv',
                     sep='\t', usecols=['category', 'headline', 'text'])
    if use_article_text:
        df['headline'] = df['headline'] + '\n' + df['text']
    return df.drop(columns='text')


def load_validation_data(language: str, use_article_text: bool):
    assert language in get_available_languages(), 'Requested language is not available'
    df = pd.read_csv(f'../../data/{language}/dev.tsv', sep='\t',
                     usecols=['category', 'headline', 'text'])
    if use_article_text:
        df['headline'] = df['headline'] + '\n' + df['text']
    return df.drop(columns='text')


def load_test_data(language: str, use_article_text: bool):
    assert language in get_available_languages(), 'Requested language is not available'
    df = pd.read_csv(f'../../data/{language}/test.tsv', sep='\t',
                     usecols=['category', 'headline', 'text'])
    if use_article_text:
        df['headline'] = df['headline'] + '\n' + df['text']
    return df['headline'].tolist(), df['category'].tolist()


def pandas_to_dataset(df: pd.DataFrame, labels):
    return Dataset.from_pandas(df, features=Features({'headline': Value('string'), 'category': ClassLabel(names=labels)}))


def get_available_languages():
    return sorted(list(filter(lambda x: x != '.DS_Store', list(os.listdir('../../data')))))


def get_labels(language: str):
    with open(f'../../data/{language}/labels.txt', 'r') as f:
        labels = f.readlines()
    return [label.replace('\n', '') for label in labels]


def get_samples_per_class(df: pd.DataFrame, n: int):
    return df.groupby('category', as_index=False).apply(lambda cat: cat.sample(n, random_state=42)).reset_index(drop=True)


if __name__ == '__main__':
    use_article_text = True
    RUN_NUMBER = 1

    if not os.path.exists(f'results_run_{RUN_NUMBER}'):
        os.mkdir(f'results_run_{RUN_NUMBER}')

    for language in get_available_languages():
        print(f'Processing the language: {language}')
        with open(f'logfile_{RUN_NUMBER}.log', 'a+') as f:
            f.write(f'{language}\n')
        labels = get_labels(language)
        for n_samples_per_class in [5, 10, 20, 50]:
            test_input, test_gt = load_test_data(language, use_article_text)
            train_ds = pandas_to_dataset(get_samples_per_class(
                load_train_data(language, use_article_text),
                n_samples_per_class,
            ), labels)
            model = None
            trainer = None
            class_labels = ClassLabel(names=labels)
            model = SetFitModel.from_pretrained("sentence-transformers/LaBSE")
            train_dataset = sample_dataset(train_ds, label_column="category")

            trainer = SetFitTrainer(
                model=model,
                train_dataset=train_dataset,
                loss_class=CosineSimilarityLoss,
                metric="f1",
                batch_size=1,
                num_iterations=5,
                seed=42,
                num_epochs=1,
                column_mapping={"headline": "text", "category": "label"}
            )
            trainer.train()
            test_gt = [class_labels.str2int(e) for e in test_gt]
            preds = trainer.model(test_input)
            f1_metric = evaluate.load("f1")
            results = f1_metric.compute(
                predictions=preds, references=test_gt, average="weighted")
            print(f'{language}/{n_samples_per_class}: {results}')
            with open(f'logfile_{RUN_NUMBER}.log', 'a+') as f:
                f.write(f'{n_samples_per_class}: {results}\n')
