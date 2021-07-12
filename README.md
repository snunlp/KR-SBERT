# KR-SBERT

A pretrained Korean-specific [Sentence-BERT](https://github.com/UKPLab/sentence-transformers) model (Reimers and Gurevych 2019) developed by [Computational Linguistics Lab](http://knlp.snu.ac.kr/) at Seoul National University.

## How to use the KR-SBERT model in Python

### Download

```bash
$ git lfs clone https://github.com/snunlp/KR-SBERT.git
```

You may need to install [git-lfs](https://git-lfs.github.com/) first.

### Usage
We recommend Python 3.6 or higher, [scikit-learn v0.23.2](https://scikit-learn.org/stable/install.html) or higher and [sentence-transformers v0.4.1](https://github.com/UKPLab/sentence-transformers) or higher.

```python
>>> from sentence_transformers import SentenceTransformer
>>> from sklearn.metrics.pairwise import cosine_similarity
>>> model = SentenceTransformer('KR-SBERT/KR-SBERT-V40K-klueNLI-augSTS') # or your file path
>>> sentences = ['잠이 옵니다', '졸음이 옵니다', '기차가 옵니다']
>>> vectors = model.encode(sentences) # encode sentences into vectors
>>> similarities = cosine_similarity(vectors) # compute similarity between sentence vectors
>>> print(similarities)
[[0.9999999  0.65774536 0.27321893]
 [0.65774536 1.         0.2729605 ]
 [0.27321893 0.2729605  1.0000002 ]]

```

You can see the sentence '잠이 옵니다' is more similar to '졸음이 옵니다' (cosine similarity 0.65774536) than '기차가 옵니다' (cosine similarity 0.27321893).


## Model description

### Pre-trained BERT model

Using a pre-trained BERT model, a sentence is segmented into WordPiece tokens, of which contextualized output vectors are mean-pooled into a single sentence vector. We use KR-BERT-V40K, a variant of [KR-BERT](https://github.com/snunlp/KR-BERT).

### Fine-tuning SBERT model

Two sentence vectors are fed into a classifier for fine-tuning. Then, a siamese network trains the BERT weight parameters to reflect the relation between the two sentences. Finally, we fine-tune the SBERT model on the [KLUE-NLI](https://github.com/KLUE-benchmark/KLUE) dataset and the augmented [KorSTS](https://github.com/kakaobrain/KorNLUDatasets) dataset.

![](https://raw.githubusercontent.com/snunlp/KR-SBERT/main/sbert-siamese-draw-2.png)

### Augmenting KorSTS dataset

We augment the KorSTS dataset using the [in-domain straregy](https://www.sbert.net/examples/training/data_augmentation/README.html) proposed by [Thakur et al. (2021)](https://aclanthology.org/2021.naacl-main.28/).

![](https://raw.githubusercontent.com/snunlp/KR-SBERT/main/sbert-augmentation-draw.png)

## Test datasets

+ Paraphrase pair detection ([download](https://drive.google.com/file/d/1trEt1QcRG2XLxMqf0ZwIVIZGDweC3VdM/view?usp=sharing))
    + source: National Institute of Korean Language, [모두의 말뭉치](https://corpus.korean.go.kr/).

```python
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('KR-BERT/KR-SBERT-V40K-klueNLI-augSTS')
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


|Model|Accuracy|
|-|-|
|KR-SBERT-Medium-NLI-STS|0.8400|
|KR-SBERT-V40K-NLI-STS|0.8400|
|KR-SBERT-V40K-NLI-augSTS|0.8511|
|KR-SBERT-V40K-klueNLI-augSTS|**0.8628**|

## Dependencies

+ `sentence_transformers` (https://github.com/UKPLab/sentence-transformers)
