# KR-SBERT

A pretrained Korean-specific [Sentence-BERT](https://github.com/UKPLab/sentence-transformers) model developed by [Computational Linguistics Lab](http://knlp.snu.ac.kr/) at Seoul National University.

## Model

+ KR-SBERT pytorch model ([download](https://drive.google.com/file/d/1QUyp6p0bHzjz1-WSGvdcNxfAAbXxdD7R/view?usp=sharing))

### 0_Transformer

KR-BERT-MEDIUM https://github.com/snunlp/KR-BERT-MEDIUM

### 1_Pooling

768 dimensions, mean-pooling

## How to download the KR-SBERT model in Linux or MacOS

Clone this repository and download the pytorch model file from Google Drive.

```bash
$ git clone https://github.com/snunlp/KR-SBERT
$ curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1QUyp6p0bHzjz1-WSGvdcNxfAAbXxdD7R" > /dev/null
$ curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1QUyp6p0bHzjz1-WSGvdcNxfAAbXxdD7R" -o ./pytorch_model.bin
$ mv ./pytorch_model.bin ./KR-SBERT/kr-sbert/0_Transformer/pytorch_model.bin
```

## How to use the KR-SBERT model in Python

We recommend Python 3.6 or higher, [scikit-learn v0.23.2](https://scikit-learn.org/stable/install.html) or higher and [sentence-transformers v0.4.1] or higher.

```python
>>> from sentence_transformers import SentenceTransformer
>>> from sklearn.metrics.pairwise import cosine_similarity
>>> model = SentenceTransformer('KR-SBERT/kr-sbert') # or your file path
>>> sentences = ['잠이 옵니다', '졸음이 옵니다', '기차가 옵니다']
>>> vectors = model.encode(sentences) # encode sentences into vectors
>>> similarities = cosine_similarity(vectors) # compute similarity between sentence vectors
>>> print(similarities)
[[1.         0.82256407 0.10487476]
 [0.82256407 0.9999997  0.1929377 ]
 [0.10487476 0.1929377  1.        ]]

```

You can see the sentence '잠이 옵니다' is more similar to '졸음이 옵니다' (cosine similarity 0.82256407) than '기차가 옵니다' (cosine similarity 0.10487476).

## Test datasets

+ Paraphrase pair detection ([download](https://drive.google.com/file/d/1trEt1QcRG2XLxMqf0ZwIVIZGDweC3VdM/view?usp=sharing))
    + source: National Institute of Korean Language, [모두의 말뭉치](https://corpus.korean.go.kr/).

```python
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('KR-BERT/kr-sbert')
data = pd.read_csv('path/to/the/dataset')

vec1 = model.encode(data['sent1'], show_progress_bar=True, batch_size=32)
vec2 = model.encode(data['sent2'], show_progress_bar=True, batch_size=32)

data['similarity'] = [
    cosine_similarity([sent1, sent2])[0][1]
    for sent1, sent2 in tqdm(zip(vec1, vec2), total=len(data))
]

data[['paraphrase', 'similarity']].to_csv('./paraphrase-results.csv')

```

## Application for document classification

Tutorial in Google Colab: https://colab.research.google.com/drive/1xwn0gFMUhUeGaVrAOPwApyVyFfsUm5zD

## Dependencies

+ `sentence_transformers` (https://github.com/UKPLab/sentence-transformers)
