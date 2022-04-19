## [YOSM: A new Yorùbá Sentiment Corpus for Movie Reviews](https://openreview.net/forum?id=rRzx5qzVIb9)

This repository contains the code for [training movie review sentiment classification](https://github.com/IyanuSh/YOSM/tree/main/train_textclass.py) and the [YOSM data](https://github.com/IyanuSh/YOSM/tree/main/) for Yorùbá language. 

The code is based on HuggingFace implementation (License: Apache 2.0).

The license of data is in [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

### Required dependencies
* python
  * [transformers](https://pypi.org/project/transformers/) : state-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch.
  * [sklearn](https://scikit-learn.org/stable/install.html) : for F1-score evaluation
  * [ptvsd](https://pypi.org/project/ptvsd/) : remote debugging server for Python support in Visual Studio and Visual Studio Code.

```bash
pip install transformers scikit-learn ptvsd
```

If you make use of this dataset, please cite us:

### BibTeX entry and citation info
```
@article{10.1162/tacl_a_00416,
    author = {Shode, Iyanuoluwa and Adelani, David Ifeoluwa and Feldman, Anna},
    title = "{YOSM: A new Yorùbá Sentiment Corpus for Movie Reviews}",
    journal = {AfricaNLP 2022 @ICLR},
    year = {2022},
    month = {4},
    url = {https://openreview.net/forum?id=rRzx5qzVIb9},
}
```


