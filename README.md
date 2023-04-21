
# MasakhaNEWS: News Topic Classification for African Languages

[paper](https://arxiv.org/abs/2304.09972) |  [dataset](https://huggingface.co/datasets/masakhane/masakhanews)

>African languages are severely under-represented in NLP research due to lack of datasets covering several NLP tasks. While there are individual language specific datasets that are being expanded to different tasks, only a handful of NLP tasks (e.g. named entity recognition and machine translation) have standardized benchmark datasets covering several geographical and typologically-diverse African languages. In this paper, we develop MasakhaNEWS -- a new benchmark dataset for news topic classification covering 16 languages widely spoken in Africa. We provide an evaluation of baseline models by training classical machine learning models and fine-tuning several language models. Furthermore, we explore several alternatives to full fine-tuning of language models that are better suited for zero-shot and few-shot learning such as cross-lingual parameter-efficient fine-tuning (like MAD-X), pattern exploiting training (PET), prompting language models (like ChatGPT), and prompt-free sentence transformer fine-tuning (SetFit and Cohere Embedding API). Our evaluation in zero-shot setting shows the potential of prompting ChatGPT for news topic classification in low-resource African languages, achieving an average performance of 70 F1 points without leveraging additional supervision like MAD-X. In few-shot setting, we show that with as little as 10 examples per label, we achieved more than 90\% (i.e. 86.0 F1 points) of the performance of full supervised training (92.6 F1 points) leveraging the PET approach.

## Languages
There are 16 languages available :
    
    - Amharic (amh)
    - English (eng)
    - French (fra)
    - Hausa (hau)
    - Igbo (ibo)
    - Lingala (lin)
    - Luganda (lug)
    - Oromo (orm)
    - Nigerian Pidgin (pcm)
    - Rundi (run)
    - chShona (sna)
    - Somali (som)
    - Kiswahili (swą)
    - Tigrinya (tir)
    - isiXhosa (xho)
    - Yorùbá (yor)



### Data Splits

For all languages, there are three splits.

The original splits were named `train`, `dev` and `test` and they correspond to the `train`, `validation` and `test` splits.

The splits have the following sizes :

| Language        | train | validation | test |
|-----------------|------:|-----------:|-----:|
| Amharic         |  1311 |        188 |  376 |
| English         |  3309 |        472 |  948 |
| French          |  1476 |        211 |  422 |
| Hausa           |  2219 |        317 |  637 |
| Igbo            |  1356 |        194 |  390 |
| Lingala     	  |   608 |         87 |  175 |
| Luganda         |   771 |        110 |  223 |
| Oromo           |  1015 |        145 |  292 |
| Nigerian-Pidgin |  1060 |        152 |  305 |
| Rundi           |  1117 |        159 |  322 |
| chiShona        |  1288 |        185 |  369 |
| Somali          |  1021 |        148 |  294 |
| Kiswahili       |  1658 |        237 |  476 |
| Tigrinya        |   947 |        137 |  272 |
| isiXhosa        |  1032 |        147 |  297 |
| Yoruba          |  1433 |        206 |  411 |



## Data Usage 
```
from datasets import load_dataset
data = load_dataset('masakhanews', 'yor') 

# Please, specify the language code

# A data point example is below:

{
'label': 0, 
'headline': "'The barriers to entry have gone - go for it now'", 
'text': "j Lalvani, CEO of Vitabiotics and former Dragons' Den star, shares his business advice for our CEO Secrets series.\nProduced, filmed and edited by Dougal Shaw", 
'headline_text': "'The barriers to entry have gone - go for it now' j Lalvani, CEO of Vitabiotics and former Dragons' Den star, shares his business advice for our CEO Secrets series.\nProduced, filmed and edited by Dougal Shaw", 
'url': '/news/business-61880859'
}
```




### BibTeX entry and citation info
```
@inproceedings{Adelani2023MasakhaNEWSNT,
  title={MasakhaNEWS: News Topic Classification for African languages},
  author={David Ifeoluwa Adelani and Marek Masiak and Israel Abebe Azime and Jesujoba Oluwadara Alabi and Atnafu Lambebo Tonja and Christine Mwase and Odunayo Ogundepo and Bonaventure F. P. Dossou and Akintunde Oladipo and Doreen Nixdorf and Chris Chinenye Emezue and Sana Al-Azzawi and Blessing K. Sibanda and Davis David and Lolwethu Ndolela and Jonathan Mukiibi and Tunde Oluwaseyi Ajayi and Tatiana Moteu Ngoli and Brian Odhiambo and Abraham Toluwase Owodunni and Nnaemeka C. Obiefuna and Shamsuddeen Hassan Muhammad and Saheed Salahudeen Abdullahi and Mesay Gemeda Yigezu and Tajuddeen Rabiu Gwadabe and Idris Abdulmumin and Mahlet Taye Bame and Oluwabusayo Olufunke Awoyomi and Iyanuoluwa Shode and Tolulope Anu Adelani and Habiba Abdulganiy Kailani and Abdul-Hakeem Omotayo and Adetola Adeeko and Afolabi Abeeb and Anuoluwapo Aremu and Olanrewaju Samuel and Clemencia Siro and Wangari Kimotho and Onyekachi Raphael Ogbu and Chinedu E. Mbonu and Chiamaka I. Chukwuneke and Samuel Fanijo and Jessica Ojo and Oyinkansola F. Awosan and Tadesse Kebede Guge and Sakayo Toadoum Sari and Pamela Nyatsine and Freedmore Sidume and Oreen Yousuf and Mardiyyah Oduwole and Ussen Abre Kimanuka and Kanda Patrick Tshinu and Thina Diko and Siyanda Nxakama and Abdulmejid Tuni Johar and Sinodos Gebre and Muhidin Mohamed and S. A. Mohamed and Fuad Mire Hassan and Moges Ahmed Mehamed and Evrard Ngabire and Pontus Stenetorp},
  year={2023}
}

```
