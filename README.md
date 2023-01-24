## Adapting [YOSM](https://openreview.net/forum?id=rRzx5qzVIb9) for MasakhaNEWS training

The YOSM repository contains the code for [training movie review sentiment classification](https://github.com/IyanuSh/YOSM/tree/main/train_textclass.py) and the [YOSM data](https://github.com/IyanuSh/YOSM/tree/main/data/yosm) for Yorùbá language. To run the code, see any of the bash scripts (*.sh)



### for `bert-base-multilingual-cased`
1. Tweak `run.sh `.
2. then run `bash mbert.sh`

Changes I made:
1. added a `args.headline_use` which determines whether to take just the headline or the headline + text as the text.