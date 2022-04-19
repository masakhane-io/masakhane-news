import pandas as pd
from datasets import load_dataset
import unicodedata
import string
import os
from nltk.tokenize import word_tokenize
import sacrebleu
from mosestokenizer import *

def create_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def download_imdb():
    imdb_train = load_dataset('imdb', split='train')
    imdb_test = load_dataset('imdb', split='test')

    #imdb_train_csv = imdb_train.to_csv()

    imdb_train.to_csv("data/imdb/train.csv")
    imdb_test.to_csv("data/imdb/test.csv")

    print(imdb_train.shape)

def standardize_label(df):
    df['text'] = df['text'].replace('\n', ' ', regex=True)
    df['label'] = df['label'].apply({0: 'negative', 1: 'positive'}.get)

    return df

def preprocess_imdb():
    tdf = pd.read_csv('data/imdb/train.csv')
    tdf = tdf.sample(frac=1, random_state=2022)
    print(tdf.shape)
    train_df = standardize_label(tdf.iloc[:20000])
    val_df = standardize_label(tdf.iloc[20000:])
    print(val_df.shape)

    test_df = pd.read_csv('data/imdb/test.csv')
    test_df = standardize_label(test_df)

    train_df[['text','label']].to_csv('data/imdb/train.tsv', sep='\t', index=None)
    val_df[['text', 'label']].to_csv('data/imdb/dev.tsv', sep='\t', index=None)
    test_df[['text', 'label']].to_csv('data/imdb/test.tsv', sep='\t', index=None)


def preprocess_yosm():
    df_1 = pd.read_csv('data/yosm/dataset.csv', nrows=1475)
    df_1 = df_1.replace('\n', ' ', regex=True)
    print(df_1.tail())
    df_1 = df_1.sample(frac=1, random_state=2022)

    df_new = pd.read_csv('data/yosm/dataset.csv', nrows=1500)
    df_new = df_new.replace('\n', ' ', regex=True)
    df_new = df_new.iloc[1475:]

    df = pd.concat([df_1, df_new], axis=0)
    print(df.shape)



    df_pos = df.loc[df['sentiment'] == 'positive']
    df_neg = df.loc[df['sentiment'] == 'negative']

    print(df_pos.shape, df_neg.shape)

    df_test = pd.concat([df_pos.iloc[:250], df_neg.iloc[:250]], axis=0) #500
    df_train = pd.concat([df_pos.iloc[250:650], df_neg.iloc[250:650]], axis=0) #800
    df_dev = pd.concat([df_pos.iloc[650:], df_neg.iloc[650:]], axis=0)  # remaining

    yo_yo_dir = 'data/yosm/yosm/'
    create_dir(yo_yo_dir)
    df_test[['yo_review','sentiment']].to_csv(yo_yo_dir+'test.tsv', sep='\t', index=None)
    df_train[['yo_review', 'sentiment']].to_csv(yo_yo_dir + 'train.tsv', sep='\t', index=None)
    df_dev[['yo_review', 'sentiment']].to_csv(yo_yo_dir + 'dev.tsv', sep='\t', index=None)


    en_yo_dir = 'data/yosm/en_yo/'
    create_dir(en_yo_dir)
    df_test[['yo_review', 'sentiment']].to_csv(en_yo_dir + 'test.tsv', sep='\t', index=None)
    df_train[['en_review', 'sentiment']].to_csv(en_yo_dir + 'train.tsv', sep='\t', index=None)
    df_dev[['en_review', 'sentiment']].to_csv(en_yo_dir + 'dev.tsv',  sep='\t', index=None)


    yoMT_yo_dir = 'data/yosm/yoMT_yo/'
    create_dir(yoMT_yo_dir)
    df_test[['yo_review', 'sentiment']].to_csv(yoMT_yo_dir + 'test.tsv', sep='\t', index=None)
    df_train[['yo_mt_review', 'sentiment']].to_csv(yoMT_yo_dir + 'train.tsv', sep='\t', index=None)
    df_dev[['yo_mt_review', 'sentiment']].to_csv(yoMT_yo_dir + 'dev.tsv', sep='\t', index=None)

    enyoMT_yo_dir = 'data/yosm/enyoMT_yo/'
    create_dir(enyoMT_yo_dir)
    df_test[['yo_review', 'sentiment']].to_csv(enyoMT_yo_dir + 'test.tsv', sep='\t', index=None)
    # rename en_review and yo_mt_review
    df_train_en = df_train[['en_review', 'sentiment']]
    df_train_en.columns = ['review', 'sentiment']
    df_train_yomt = df_train[['yo_mt_review', 'sentiment']]
    df_train_yomt.columns = ['review', 'sentiment']
    # concatenate reviews translated by google translate to yoruba and english reviews
    df_train_large = pd.concat([df_train_en, df_train_yomt], axis=0)
    df_train_large.to_csv(enyoMT_yo_dir + 'train.tsv', sep='\t', index=None)
    df_dev[['yo_mt_review', 'sentiment']].to_csv(enyoMT_yo_dir + 'dev.tsv', sep='\t', index=None)


def detokenize(sentences):
    tokenized_sentences = []
    with MosesDetokenizer('en') as detokenize:
        for sent in sentences:
            #print(sent)
            tokenized_sent = detokenize(sent.split())
            tokenized_sentences.append(tokenized_sent)

    return tokenized_sentences

def compute_bleu():
    df = pd.read_csv('data/yosm/dataset.csv', nrows=1500)
    df = df.replace('\n', ' ', regex=True)
    human = list(df['yo_review'].values)
    mt = list(df['yo_mt_review'].values)

    print(len(mt), len(human))
    bleu_sacre = sacrebleu.corpus_bleu(mt, [human])
    print('BLEU score: ', bleu_sacre.score)

if __name__ == "__main__":
    #download_imdb()
    #preprocess_imdb()
    preprocess_yosm()
    compute_bleu()
