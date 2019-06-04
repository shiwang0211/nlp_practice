# NLP Practice
The purpose of this repository is to practice NLP-related concepts and models 

## 1. Concepts and Fundamentals
### [Concept Notes.ipynb](https://github.com/shiwang0211/nlp_practice/blob/master/Concept√Notes.ipynb)
- Basics of English Grammar
    - Context-free Grammer (CFG)
- Text Normalization
- Parsing
    - Syntactic parsing and CKY Algorithm
    - Statistical parsing and Probabilistic CFG
    - Dependency Parsing
- Word Vectorization
    - Bag-of-Word and TF-IDF
    - Word-Embedding and Word2Vec
- Language Modelling
    - N-gram Model
    - Neural Network Models
- Part-of-Speeh (PoS) Tagging 
    - Hidden-Markov-Model (HMM)
    - Maximum Entropy Markov-Model (MEMM)
- Named Entity Recognition
- Sentiment Analysis
- Word Sense Disambiguity
- Machine Translation
    - Encoder-Decoder
    - Attention model (Luong attention, Bahdanau Attention)
- Question Answering Model
- Coreference Resolution


## 2. Model Applications
### [Word Embedding.ipynb](Word_Embedding.ipynb)
- Implement back-propagation of word vectorization model with `numpy`
    - Calculation of probability
    - Define loss function
    - Calculation of gradients and Update rules
    - Validation of model using a simple example
- Implemented the **Skip-Gram** model with **Tensorflow** low-level API
- Applied off-the-shelf `Word2Vec` package to train model and generate word embeddings


### [Topic Modelling.ipynb](Topic%20Modelling.ipynb)
- Review of the maths behind Latent Dirichlet Allocation (LDA)\
- Implementation using `sklearn`


### [Sentiment_Analysis_v1.ipynb](Sentiment_Analysis_v1.ipynb)
- Replicate Sentiment Analysis (Negative/Positive Comments)
  - Use pretrained word vectors
  - Methodology: 1) LSTM; 2) CNN 
  - Tool: Tensorflow lower-level API

### [Sentiment_Analysis_v2.ipynb](Sentiment_Analysis_v2.ipynb)
- Replicate Sentiment Analysis (Negative/Positive Comments)
  - Train word vectors using `word2vec` package
  - Methodology: 1) LSTM; 2) CNN 
  - Tool: Keras with Tensorflow Backend

### [seq2seq.ipynb](seq2seq.ipynb)
- Imeplemented a character-level translation model with Encoder/Decoder Machine Translation architecture (with Keras)
    - Encoder Network (1-direction LSTM) 
    - Decoder Network (1-direction LSTM) - Teacher Forcing & Sampling
- Implemented a Encoder/Decoder model with attention
    - Added extra Luong attention mechanism on top of Encoder/Decoder model
    
    
### [yelp_nlp.ipynb](yelp_nlp.ipynb)
The following analysis is performed on the yelp dataset published at: https://www.yelp.com/dataset/challenge:
- Data preprocessing from raw json files
- Application of Phrase Model
- Natural Language Processing for review texts
    - Token filtering
    - Phase model
    - TF-IDF
    - Doc2Vec
- Collaborative-Filtering with ratings
    - User-based
    - Item-based
    - Matrix Factorization
- Content-based Filtering
    - Define Business Profile based on rating-weighted embedded review text
    - Define User Profile based on Business Profiles
    - Generate recommendation based on cosine-similarity
- Topic Modeling and Latent Dirichlet Allocation (LDA) implemented with **gensim**
- Sentiment Analysis with LSTM model implemented with **Tensorflow** and **Keras**

## 3. Popular Packages]
### [Chinese_nlp.ipynb](Chinese_nlp.ipynb)
Work through examples with `jieba` package for text:
- Jiaba word parsing, keyword extraction, PoS Tagging
- Text classification using LSTM model for a Chinese NLP problem

### [nltk_learn.ipynb](nltk_learn.ipynb)
Work through examples with `NLTK` package for text:
- Preprocessing (tokenize, stop words, stemming, etc)
- Feature extraction (DTM, TF-IDF)
- Text classification, and `scikit learn`
- PoS Tagging

### [spacy_learn.ipynb](spacy_learn.ipynb)
Work through examples with `spacy` package for text:
- Dependency-parsing, similarity, tokenizer, etc.
- Named Entity Recognition
- Vocab, Hash, Lexeme



