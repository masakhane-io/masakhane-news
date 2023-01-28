# masakhane-news
MasakhaNEWS: News Topic Classification for African Languages

- clone this branch
- Create a folder called `results`
- According to your need, modify the models and necessary information on `/home/mila/b/bonaventure.dossou/masakhane-news/code/sample.sh` otherwise you should be good to train all models, for all languages, across 5 seeds and the two headings styles (`0` is when qwe consider only the text, `1` is when we combine the heading of the article to its supposedly for situation where we need more context than the headline to label the article)
- run `bash/sbatch /home/mila/b/bonaventure.dossou/masakhane-news/code/sample.sh`