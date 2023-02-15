from src.helpers import (
    get_available_languages, get_classifications, load_train_data, load_test_data,
    get_samples_per_class, train_data_to_cohere_examples, get_co, process_classifications,
    add_metadata
)
from tqdm import tqdm
import json
import os

if __name__ == '__main__':
    use_article_text = True
    RUN_NUMBER = 1
    f1s = {
        5: [],
        10: [],
        20: [],
        50: []
    }

    if os.path.exists(f'results_run_{RUN_NUMBER}'):
        raise Exception(f'results run {RUN_NUMBER} already in use')

    os.mkdir(f'results_run_{RUN_NUMBER}')
    co = get_co()
    for language in get_available_languages():
        print(f'Processing the language: {language}')
        with open(f'logfile_{RUN_NUMBER}.log', 'a+') as f:
            f.write(f'{language}\n')
        inputs, ground_truth = load_test_data(language, use_article_text)
        for n_samples_per_class in [5, 10, 20, 50]:
            examples = train_data_to_cohere_examples(
                get_samples_per_class(
                    load_train_data(language, use_article_text),
                    n_samples_per_class,
                )
            )
            cumulative_classifications = []
            for input_samples_idx in tqdm(range(0, len(inputs), 96)):
                classifications = get_classifications(
                    co, examples, inputs[input_samples_idx:input_samples_idx+96]
                ).classifications
                cumulative_classifications.extend(classifications)
            data, accuracy, f1_score_val = process_classifications(
                inputs, ground_truth, cumulative_classifications
            )
            with open(f'logfile_{RUN_NUMBER}.log', 'a+') as f:
                f.write(
                    f'{n_samples_per_class}: accuracy ({accuracy}), f1_score ({f1_score_val})\n')
            f1s[n_samples_per_class].append(f1_score_val)
            print(
                f'{n_samples_per_class}: accuracy ({accuracy}), f1: ({f1_score_val})')
            data = add_metadata({'num_train': n_samples_per_class,
                                 'accuracy': accuracy,
                                 'f1_score': f1_score_val}, data)
            if not os.path.exists(f'results_run_{RUN_NUMBER}/{language}'):
                os.mkdir(f'results_run_{RUN_NUMBER}/{language}')
            with open(f'results_run_{RUN_NUMBER}/{language}/cohere_train_{language}_{n_samples_per_class}.out', 'w+', encoding='utf-8') as f:
                json.dump(data, f)
    for lang_list in f1s:
        for num in f1s[lang_list]:
            num *= 100
            print("%.1f" % round(num, 1), end=' & ')
        print()
    with open(f'results_run_{RUN_NUMBER}/overleaf.log', 'w+') as f:
        json.dump(f1s, f)
