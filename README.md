# NLP Practice
The purpose of this repository is to practice NLP-related concepts and models 

## 1. [Concept Notes.ipynb](https://github.com/shiwang0211/nlp_practice/blob/master/Concept%20Notes.ipynb)
- Language Modelling
    - N-gram
    - Word2Vec
- Machine Translation
    - Encoder-Decoder
    - Attention model
- Coreference Resolution

## 2. [yelp_nlp.ipynb](https://github.com/shiwang0211/nlp_practice/blob/master/yelp_nlp.ipynb)
The following analysis is performed on the yelp dataset:
- Data preprocessing from raw json files
- Collaborative-Filtering with ratings
    - User-based
    - Item-based
    - Matrix Factorization
- Natural Language Processing for review texts
    - Token filtering
    - Phase model
    - TF-IDF
    - Doc2Vec
- Content-based Filtering
    - User and Item profile
- Topic Modelling with `gensim`
- Sentiment Analysis with LSTM

## 3. [seq2seq.ipynb](https://github.com/shiwang0211/nlp_practice/blob/master/seq2seq.ipynb)
- Encoder/Decoder Machine Translation architecture with a character-level translation model (implemented with Keras)
- Develop attention model with a synthetic example 
    - Add extra attention mechanism based on Keras layers

## 4.[Word Embedding.ipynb](https://github.com/shiwang0211/nlp_practice/blob/master/Word_Embedding.ipynb)
- Implement back-propagation of word vectorization model with `numpy`
- Build word representation using `Skip-Gram` model (with Tensorflow low-level API)
    - Prepare data batch
    - Define network graph
- Use off-the-shelf `Word2Vec` package to train model

## 5. [nltk_learn.ipynb](https://github.com/shiwang0211/nlp_practice/blob/master/nltk_learn.ipynb)
Work through examples with `NLTK` package for text:
- Preprocessing (tokenize, stop words, stemming, etc)
- Feature extraction (DTM, TF-IDF)

## 6. [spacy_learn.ipynb](https://github.com/shiwang0211/nlp_practice/blob/master/spacy_learn.ipynb)
Work through examples with `spacy` package for text:
- Dependency-parsing, similarity, tokenizer, etc.
- Named Entity Recognition
- Vocab, Hash, Lexeme

## 7. [Sentiment_Analysis_v1.ipynb](https://github.com/shiwang0211/nlp_practice/blob/master/Sentiment_Analysis_v1.ipynb)
- Replicate Sentiment Analysis (Negative/Positive Comments)
  - Use pretrained word vectors
  - Methodology: 1) LSTM; 2) CNN 
  - Tool: Tensorflow lower-level API

## 8. [Sentiment_Analysis_v2.ipynb](https://github.com/shiwang0211/nlp_practice/blob/master/Sentiment_Analysis_v2.ipynb)
- Replicate Sentiment Analysis (Negative/Positive Comments)
  - Train word vectors using `word2vec` package
  - Methodology: 1) LSTM; 2) CNN 
  - Tool: Keras with Tensorflow Backend

