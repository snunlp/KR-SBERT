# KR-SBERT

A pretrained Korean-specific [Sentence-BERT](https://github.com/UKPLab/sentence-transformers) model developed by [Computational Linguistics Lab](http://knlp.snu.ac.kr/) at Seoul National University.

## Model Pre-training & Fine-tuning

KR-BERT-V40K

KLUE-NLI

KorSTS-augmented

## How to use the KR-SBERT model in Python

We recommend Python 3.6 or higher, [scikit-learn v0.23.2](https://scikit-learn.org/stable/install.html) or higher and [sentence-transformers v0.4.1](https://github.com/UKPLab/sentence-transformers) or higher.

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

data[['paraphrase', 'similarity']].to_csv('paraphrase-results.csv')

```

## Application for document classification

Tutorial in Google Colab: https://colab.research.google.com/drive/1S6WSjOx9h6Wh_rX1Z2UXwx9i_uHLlOiM

## Dependencies

+ `sentence_transformers` (https://github.com/UKPLab/sentence-transformers)
