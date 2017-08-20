
# coding: utf-8

# # 1. Neural Language Model
# If you are here that means you wish to cut the crap and understand how to train your own Neural Language Model. If you are a regular user of frameworks like Keras, Tflearn, etc., then you know how easy it has become these days to build, train and deploy Neural Network Models. If not then you will probably by the end of this post.
# 
# # 2. Prerequisite
# 1. [Python](https://www.tutorialspoint.com/python/): I will be using Python 3.5 for this tutorial
# 
# 2. [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/): If you dont know what LSTMs are, then this is a must read.
# 
# 3. [Basics of Machine Learning](https://www.youtube.com/watch?v=2uiulzZxmGg): If you want to dive into Machine Learning/Deep Learning, then I strongly recommend the first 4 lectures from [Stanford's CS231]() by Andrej Karpathy.
# 
# 4. [Language Model](https://en.wikipedia.org/wiki/Language_model): If you want to have a basic understanding of Language Models.
# 
# # 3. Frameworks
# 1. [Tflearn](http://tflearn.org/installation/) 0.3.2
# 2. [Spacy](https://spacy.io/) 1.9.0
# 3. [Tensorflow](https://spacy.io/) 1.0.1
# 
# ### Note
# you can take this post as a hands-on exercise on "How to build your own Neural Language Model" from scratch. If you have a ready to use virtualenv with all the dependencies installed then you can skip Section 4 and jump to Section 5. 

# # 4. Install Dependencies
# We will install everythin in a virtual environment and I would suggest you to run this Jupyter Notebook in the same virtualenv. I have also provided a ```requirements.txt``` file with the [repository](https://github.com/dashayushman/neural-language-model) to make things easier.
# 
# ### 4.1 Virtual Environment
# You can follow [this](http://docs.python-guide.org/en/latest/dev/virtualenvs/) for a fast guide to Virtual Environments.
# 
# ```sh
# pip install virtualenv
# ```
# 
# ### 4.2 Tflearn
# Follow [this](http://tflearn.org/installation/) and install Tflearn. Make sure to have the versions correct in case you want to avoid weird errors. 
# 
# ```sh
# pip install -Iv tflearn==0.3.2
# ```
# 
# ### 4.3 Tensorflow
# Install Tensorflow by following the instructions [here](https://www.tensorflow.org/install/). To make sure of installing the right version, use this
# 
# ```sh
# pip install -Iv tensorflow-gpu==1.0.1
# ```
# Note that this is the GPU version of Tensorflow. You can even install the CPU version for this tutorial, but I would strongly recommend the GPU version if you intend to intend to scale it to use in the real world.
# 
# ### 4.4 Spacy
# Install Spacy by following the instructions [here](https://spacy.io/docs/usage/). For the right version use,
# 
# ```sh
# pip install -Iv spacy==1.9.0
# ```
# 
# ### 4.5 Others
# ```sh
# pip install numpy
# ```

# # 5. Get the Repo
# clone the Neural Language Model GitHub repository onto your computer and start the Jupyter Notebook server.
# 
# ```sh
# git clone https://github.com/dashayushman/neural-language-model.git
# cd neural-language-model
# jupyter notebook
# ```
# 
# Open the notebook names **Neural Language Model** and you can start off.

# # 6. Neural Language Model
# We will start building our own Language model using an LSTM Network. To do so we will need a corpus. For the purpose of this tutorial, let us use a toy corpus, which is a text file called ```corpus.txt``` that 0I downloaded from Wikipedia. I will use this to demponstrate how to build your own Neural Language Model, and you can use the same knowledge to extend the model further for a more realistic scenario (I will give pointers to do so too).
# 
# ## 6.1 Loading The Corpus
# In this section you will load the ```corpus.txt``` and do minimal preprocessing.

# In[1]:


import re

with open('corpus.txt', 'r') as cf:
    corpus = []
    for line in cf: # loops over all the lines in the corpus
        line = line.strip() # strips off \n \r from the ends 
        if line: # Take only non empty lines
            line = re.sub(r'\([^)]*\)', '', line) # Regular Expression to remove text in between brackets
            line = re.sub(' +',' ', line) # Removes consecutive spaces
            # add more pre-processing steps
            corpus.append(line)
print("\n".join(corpus[:5])) # Shows the first 5 lines of the corpus


# As you can see that this small piece of code loads the toy text corpus, extracts lines from it, ignores empty lines, and removes text in between brackets. Note that in reality you will not be able to load the entire corpus into memory. You will need to write a [generator](https://wiki.python.org/moin/Generators) to yield text lines from the corpus, or use some advanced features provided by the Deep Learning frameworks like [Tensorflow's Input Pipelines](https://www.tensorflow.org/programmers_guide/reading_data). 
# 
# ## 6.2 Tokenizing the Corpus
# In this section we will see how to tokenize the text lines that we extracted and then create a **Vocabulary**.

# In[2]:


# Load Spacy
import spacy
import numpy as np
nlp = spacy.load('en_core_web_sm')


# In[3]:


def preprocess_corpus(corpus):
    corpus_tokens = []
    sentence_lengths = []
    for line in corpus:
        doc = nlp(line) # Parse each line in the corpus
        for sent in doc.sents: # Loop over all the sentences in the line
            corpus_tokens.append('SEQUENCE_BEGIN')
            s_len = 1
            for tok in sent: # Loop over all the words in a sentence
                if tok.text.strip() != '' and tok.ent_type_ != '': # If the token is a Named Entity then do not lowercase it 
                    corpus_tokens.append(tok.text)
                else:
                    corpus_tokens.append(tok.text.lower())
                s_len += 1
            corpus_tokens.append('SEQUENCE_END')
            sentence_lengths.append(s_len+1)
    return corpus_tokens, sentence_lengths

corpus_tokens, sentence_lengths = preprocess_corpus(corpus)
print(corpus_tokens[:30]) # Prints the first 30 tokens
mean_sentence_length = np.mean(sentence_lengths)
deviation_sentence_length = np.std(sentence_lengths)
max_sentence_length = np.max(sentence_lengths)
print('Mean Sentence Length: {}\nSentence Length Standard Deviation: {}\n'
      'Max Sentence Length: {}'.format(mean_sentence_length, deviation_sentence_length, max_sentence_length))


# Notice that we did not lowercase the [Named Entities(NEs)](https://en.wikipedia.org/wiki/Named-entity_recognition). This is totally your choice. It part of a normalization step and I believe it is a good idea to let the model learn the Named Entities in the corpus. But do not blindly consider any library for NEs. I chose Spacy as it is very simple to use, fast and efficient. Note that I am using the [**en_core_web_sm**](https://spacy.io/docs/usage/models) model of Spacy, which is very small and good enough for this tutorial. You would probably want to choose your own NE recognizer.
# 
# Other Normalization steps include [stemming and lemmatization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html) which I will not implement because **(1)** I want my Language Model to learn the various forms of a word and their occurances by itself; **(2)** In a real world scenario you will train your Model with a huge corpus with Millions of text lines, and you can assume that the corpus covers the most commonly used terms in Language. Hence, no extra normalization is required. 
# 
# ### 6.2.1 SEQUENCE_BEGIN and SEQUENCE_END
# Along with the naturally occurring terms in the corpus, we will add two new terms called the *SEQUENCE_BEGIN* and **SEQUENCE_END** term. These terms mark the beginning and end of a sentence. We do this because we want our model to learn word occurring at the beginning and at the end of sentences. Note that we are dependent on Spacy's Tokenization algorithm here. You are free to explore other tokenizers and use whichever you find is best.
# 
# ## 6.3 Create a Vocabulary
# After we have minimally preprocessed the corpus and extracted sequence of terms from it, we will create a vocabulary for our Language Model. This means that we will create two python dictionaries,
# 1. **Word2Idx** : This dictionary has all the unique words(terms) as keys with a corresponding unique ID as values
# 2. **Idx2Word** : This is the reverse of Word2Idx. It has the unique IDs as keys and their corresponding words(terms) as values

# In[4]:


vocab = list(set(corpus_tokens)) # This works well for a very small corpus
#print(vocab)


# **Alternatively**, if your corpus is huge, you would probably want to iterate through it entirely and generate term frequencies. Once you have the term frequencies, it is better to select the most commonly occuring terms in the vocabulary (as it covers most of the Natural Language).

# In[5]:


import collections

word_counter = collections.Counter()
for term in corpus_tokens:
    word_counter.update({term: 1})
vocab = word_counter.most_common(10000) # 10000 Most common terms
print('Vocab Size: {}'.format(len(vocab))) 
print(word_counter.most_common(100)) # just to show the top 100 terms


# This was we make sure to consider the ***top K***(in this case 100) most commonly used terms in the Language (assuming that the corpus represents the Language or domain specific language. For e.g., medical corpora, e-commerce corpora, etc.). In Neural Machine Translation Models, usually a vocabulary size of 10,000 to 100,000 is used. But remember, it all depends on your task, corpus size, and the Language itself. 

# ### 6.3.1 UNKNOWN and PAD
# Along with the vocabulary terms that we generated, we need two more special terms:
# 1. **UNKNOWN**: This term is used for all the words that the model will observe apart from the vocabulary terms.
# 2. **PAD**: The pad term is used to pad the sequences to a maximum length. This is required for feeding variable length sequences into the Network (we use DynamicRnn to handle variable length sequences. So, padding makes no difference. It is just required for feeding the data to Tensorflow)
# 
# This is required as during inference time there will be many unknown words (words that the model has never seen). It is better to add an **UNKNOWN** token in the vocabulary so that the model will learn to handle terms that are unknown to the Model.

# In[6]:


vocab.append(('UNKNOWN', 1))
Idx = range(1, len(vocab)+1)
vocab = [t[0] for t in vocab]

Word2Idx = dict(zip(vocab, Idx))
Idx2Word = dict(zip(Idx, vocab))

Word2Idx['PAD'] = 0
Idx2Word[0] = 'PAD'
VOCAB_SIZE = len(Word2Idx)
print('Word2Idx Size: {}'.format(len(Word2Idx)))
print('Idx2Word Size: {}'.format(len(Idx2Word)))


# ## 6.4 Preload Word Vectors
# Since you are here, I am almost sure that you are familiar with or have atleast heard of [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html). Read about it if you don't know. 
# 
# Spacy provides a set of pretrained word vectors. We will make use of these to initialize our embedding layer (details in the following section). 

# In[7]:


w2v = np.random.rand(len(Word2Idx), 300) # We use 300 because Spacy provides us with vectors of size 300

for w_i, key in enumerate(Word2Idx):
    token = nlp(key)
    if token.has_vector:
        #print(token.text, Word2Idx[key])
        w2v[Word2Idx[key]:] = token.vector
EMBEDDING_SIZE = w2v.shape[-1]
print('Shape of w2v: {}'.format(w2v.shape))
print('Some Vectors')
print(w2v[0][:10], Idx2Word[0])
print(w2v[80][:10], Idx2Word[80])


# ## 6.5 Splitting the Data
# We are almost there. Have patience :) We need to split the data into Training and Validation set before we proceed any further. So,

# In[8]:


train_val_split = int(len(corpus_tokens) * 0.8) # We use 80% of the data for Training and 20% for validating
train = corpus_tokens[:train_val_split]
validation = corpus_tokens[train_val_split:-1]

print('Train Size: {}\nValidation Size: {}'.format(len(train), len(validation)))


# ## 6.6 Prepare The Training Data
# We will prepare the data by doing the following fro both train and Validation data:
# 1. Convert word sequences to id sequences (which will be later used in the embedding layer)
# 2. Generate n-grams from the input sequences
# 3. Pad the generated n_grams to a max-length so that it can be fed to Tensorflow

# In[9]:


from tflearn.data_utils import to_categorical, pad_sequences


# In[10]:


# A method to convert a sequence of words into a sequence of IDs given a Word2Idx dictionary
def word2idseq(data, word2idx):
    id_seq = []
    for word in data:
        if word in word2idx:
            id_seq.append(word2idx[word])
        else:
            id_seq.append(word2idx['UNKNOWN'])
    return id_seq

# Thanks to http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
# This method generated n-grams
def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

train_id_seqs = word2idseq(train, Word2Idx)
validation_id_seqs = word2idseq(validation, Word2Idx)

print('Sample Train IDs')
print(train_id_seqs[-10:-1])
print('Sample Validation IDs')
print(validation_id_seqs[-10:-1])


# ### 6.6.1 Generating the Targets from N-Grams
# This might look a little tricky but it is not. Here we take the sequence of ids and generate n-grams. For the purpose of training, we need sequences of terms as the training examples and the next term in the sequence as the target. Not clear right? Let us look at an example. If our sequence of words were ```['hello', 'my', 'friend']```, then we extract extract n-grams, where n=2-3 (that means we split bigrams and trigrams from the sequence). So the sequence is split into ```['hello', 'my'], ['my', 'friend'] and ['hello', 'my', 'friend']```. Well to train our network this is not enough right? We need some objective/target that we can infer about. So to get a target we split the last term of the n-grams out. In the case of our example, the corresponding targets are ```['friend', 'my', 'friend']```. To show you the bigger picture, the input sequence ```['my', 'friend', 'friend']``` is split into n-grams and then split again to pop out a target term.
# 
# ```python
# bigram['hello', 'my'] --> input['hello'] --> target['my']
# bigram['my', 'friend'] --> input['my'] --> target['friend']
# trigram['hello', 'my', 'friend'] --> input['hello', 'my'] --> target['friend']
# ```

# In[11]:


import random

def prepare_data(data, n_grams=5, batch_size=64, n_epochs=10):
    X, Y = [], []
    buff_size, start, end = 1000, 0, 1000
    n_buffer = 0
    epoch = 0
    while epoch < n_epochs:
        if len(X) >= batch_size:
            X_batch = X[:batch_size]
            Y_batch = Y[:batch_size]
            X_batch = pad_sequences(X_batch, maxlen=n_grams, value=0)
            Y_batch = to_categorical(Y_batch, VOCAB_SIZE)
            yield (X_batch, Y_batch, epoch)
            X = X[batch_size:]
            Y = Y[batch_size:]
            continue
        n = random.randrange(2, n_grams)
        if len(data) < n: continue
        if end > len(data): end = len(data)
        grams = find_ngrams(data[start: end], n) # generates the n-grams
        splits = list(zip(*grams)) # split it
        X += list(zip(*splits[:len(splits)-1])) # from the inputs
        X = [list(x) for x in X] 
        Y += splits[-1] # form the targets
        if start + buff_size > len(data):
            start = 0
            epoch += 1
            end = start + buff_size
        else:
            start = start + buff_size
            end = end + buff_size


# ## 6.7 The Model
# We now define a Dynamic LSTM Model that will be our Language Model. Restart the kernel and run all cells if it does not work (some Tflearn bug). 

# In[12]:


# Hyperparameters
LR = 0.0001
HIDDEN_DIMS = 256
N_LAYERS = 3
BATCH_SIZE = 10000
N_EPOCHS=100
N_GRAMS = 5
N_VALIDATE = 3000


# In[13]:


train = prepare_data(train_id_seqs, N_GRAMS, BATCH_SIZE, N_EPOCHS)
validate = prepare_data(validation_id_seqs, N_GRAMS, N_VALIDATE, N_EPOCHS)


# In[14]:


import tensorflow as tf
import tflearn


# In[15]:


# Build the model
embedding_matrix = tf.constant(w2v, dtype=tf.float32)
net = tflearn.input_data([None, N_GRAMS], dtype=tf.int32, name='input')
net = tflearn.embedding(net, input_dim=VOCAB_SIZE, output_dim=EMBEDDING_SIZE,
                        weights_init=embedding_matrix, trainable=True)
net = tflearn.lstm(net, HIDDEN_DIMS, dropout=0.8, dynamic=True)
net = tflearn.fully_connected(net, VOCAB_SIZE, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='target')
model = tflearn.DNN(net, best_checkpoint_path="./best_chkpnts/",
                    max_checkpoints= 100, tensorboard_dir='./chkpnts/',
                    best_val_accuracy=0.70, tensorboard_verbose=0)

prev_epoch = -1
n_batch = 1
for batch in train:
    if batch[2] != prev_epoch:
        n_batch = 1
        prev_epoch = batch[2]
        print('Training Epoch {}'.format(batch[2]))
        (X_test, Y_test, val_epoch) = next(validate)
    print('Fitting Batch: {}'.format(n_batch))
    model.fit(batch[0], batch[1], validation_set=(X_test, Y_test),
              show_metric=True, n_epoch=1)
    n_batch += 1


# # 7. Inference
# The story does not get over after you train the model. We need to understand how to make inference using this trained model. Well honestly, this model is not even close to trained. We used just one article from Wikipedia to train this Language Model so we cannot expect it to be good. The idea was to realise the steps required actually build a Language Model from scratch. Now let us look at how to make an inference from the model that we just trained.
# 
# ## 7.1 Log Probability of a Sequence 
# Given a new sequence of terms, we would like to know the probability of the occurance of this sequence in the Language. We make use of our trained model (which we assume to be a represenattion of the Langauge) and calculate the n-gram probabilities and aggregate them to find a final probability score.

# In[ ]:


def get_sequence_prob(in_string, n, model):
    in_tokens, in_lengths = preprocess_corpus(in_string)
    in_ids = word2idseq(in_tokens, Word2Idx)
    X, Y_, Y = prepare_data(in_ids, n)
    preds = model.predict(X)
    log_prob = 0.0
    for y_i, y in enumerate(Y):
        log_prob += np.log(preds[y_i, y])

    log_prob = log_prob/len(Y)
    return log_prob

in_strings = ['hello I am science', 'blah blah blah', 'deep learning', 'answer',
              'Boltzman', 'from the previous layer as input', 'ahcblheb eDHLHW SLcA']
for in_string in in_strings:
    log_prob = get_sequence_prob(in_string, 5, model)
    print(log_prob)


# To get the probability of the sequence, we take the n-grams of the sequence and we infer the probability of the next term to occur, take it's log and sum it with the log probabilities of all the other n-grams. The final score is the average over all. There can be other ways to look at it too. You can notmalize by n too, where n is the number of grans you considered. 

# # 7.2 Generating a Sequence
# Since we trained this Language model to predict the next term given the previous 'n' terms, we can sample sequences out of this model too. We start with a random term and feed it to the Model. The Model predicts the next term and then we concat it with our previous term and feed it again to the Model. In this way we can generate arbitarily long sequences from the Model. Let us see how this naive model generates sequences,

# In[ ]:


def generate_sequences(term, word2idx, idx2word, seq_len, n_grams, model):
    if term not in word2idx:
        idseq = [[word2idx['UNKNOWN']]]
    else:
        idseq = [[word2idx[term]]]
    for i in range(seq_len-1):
        #print(idseq)
        padded_idseq = pad_sequences(idseq, maxlen=n_grams, value=0)
        next_label = model.predict_label(padded_idseq)
        print(next_label)
        idseq[0].append(next_label[0][0])
    generated_str = []
    for id in idseq[0]:
        generated_str.append(idx2word[id])
    return ' '.join(generated_str)
    
term = 'SEENCE_BEGIN'
seq = generate_sequences(term, Word2Idx, Idx2Word, 10, 5, model)
print(seq)