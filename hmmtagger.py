# -*- coding: utf-8 -*-
"""HMMTagger (1).ipynb


# Part-Of-Speech-tagging

Part-Of-Speech-tagging using Hidden Markov model to identify the category of words ('noun', 'verb', ...) in plain text.

The objective is to categorize each word in sentences as one of the **12 categories of the universal POS tag set.** 
This process is called **Part-Of-Speech-Tagging.**

Part of speech tagging is the process of determining the syntactic category of a word from the words in its surrounding context. 

Parts of speech (also known as POS, word classes, or syntactic categories) are useful because they reveal a lot about a word and its neighbors. 

Knowing whether a word is a noun or a verb tells us about likely neighboring words (nouns are preceded by determiners and adjectives, verbs by nouns) and syntactic structure (nouns are generally part of noun phrases), making part-of-speech tagging a key aspect of parsing. 

It is often used to help disambiguate natural language phrases because it can be done quickly with high accuracy. 

Tagging can be used for many NLP tasks like determining correct pronunciation during speech synthesis (for example, the word content is pronounced CONtent when it is a noun and conTENT when it is an adjective, same with dis-count as a noun vs dis-count as a verb), for information retrieval, for labeling named entities like people or organizations in information extraction and for word sense disambiguation.

The universal POS tagset used in this project defines the following twelve POS tags: 
NOUN (nouns),

VERB (verbs),

ADJ (adjectives), 

ADV (adverbs),

PRON (pronouns),

DET (determiners and articles),

ADP (prepositions and postpositions),

NUM (numerals),

CONJ (conjunctions),

PRT (particles),
a conjunction is an **invariable (non-inflected) grammatical particle **and it may or may not stand between the items conjoined. 

The definition of a conjunction may also be extended to idiomatic phrases that behave as a unit with the same function,

 e.g. "as well as", "provided that".

‘.’ (punctuation marks) and 

X (a catch-all for other categories such as abbreviations or foreign words).


!pip install pomegranate

Reference: 
https://towardsdatascience.com/part-of-speech-tagging-with-hidden-markov-chain-models-e9fccc835c0e

Import the libraries and read the data.
"""

import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display import HTML
from itertools import chain
from collections import Counter, defaultdict,namedtuple,OrderedDict
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution
import random

Sentence = namedtuple("Sentence", "words tags")
def read_data(filename):
    """Read tagged sentence data"""
    with open(filename, 'r') as f:
        sentence_lines = [l.split("\n") for l in f.read().split("\n\n")]
    return OrderedDict(((s[0], Sentence(*zip(*[l.strip().split("\t")
                        for l in s[1:]]))) for s in sentence_lines if s[0]))
def read_tags(filename):
    """Read a list of word tag classes"""
    with open(filename, 'r') as f:
        tags = f.read().split("\n")
    return frozenset(tags)
Sentence = namedtuple("Sentence", "words tags")
def read_data(filename):
    """Read tagged sentence data"""
    with open(filename, 'r') as f:
        sentence_lines = [l.split("\n") for l in f.read().split("\n\n")]
    return OrderedDict(((s[0], Sentence(*zip(*[l.strip().split("\t")
                        for l in s[1:]]))) for s in sentence_lines if s[0]))
def read_tags(filename):
    """Read a list of word tag classes"""
    with open(filename, 'r') as f:
        tags = f.read().split("\n")
    return frozenset(tags)

class Subset(namedtuple("BaseSet", "sentences keys vocab X tagset Y N stream")):
    def __new__(cls, sentences, keys):
        word_sequences = tuple([sentences[k].words for k in keys])
        tag_sequences = tuple([sentences[k].tags for k in keys])
        wordset = frozenset(chain(*word_sequences))
        tagset = frozenset(chain(*tag_sequences))
        N = sum(1 for _ in chain(*(sentences[k].words for k in keys)))
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))
        return super().__new__(cls, {k: sentences[k] for k in keys}, keys, wordset, word_sequences,
                               tagset, tag_sequences, N, stream.__iter__)
def __len__(self):
        return len(self.sentences)
def __iter__(self):
        return iter(self.sentences.items())
#Dataset class
class Dataset(namedtuple("_Dataset", "sentences keys vocab X tagset Y training_set testing_set N stream")):
    def __new__(cls, tagfile, datafile, train_test_split=0.8, seed=112890):
        tagset = read_tags(tagfile)
        sentences = read_data(datafile)
        keys = tuple(sentences.keys())
        wordset = frozenset(chain(*[s.words for s in sentences.values()]))
        word_sequences = tuple([sentences[k].words for k in keys])
        tag_sequences = tuple([sentences[k].tags for k in keys])
        N = sum(1 for _ in chain(*(s.words for s in sentences.values())))
        
        # split data into train/test sets
        _keys = list(keys)
        if seed is not None: random.seed(seed)
        random.shuffle(_keys)
        split = int(train_test_split * len(_keys))
        training_data = Subset(sentences, _keys[:split])
        testing_data = Subset(sentences, _keys[split:])
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))
        return super().__new__(cls, dict(sentences), keys, wordset, word_sequences, tagset,
                               tag_sequences, training_data, testing_data, N, stream.__iter__)
def __len__(self):
        return len(self.sentences)
def __iter__(self):
        return iter(self.sentences.items())

data = Dataset("tags-universal.txt", "brown-universal.txt", train_test_split=0.8)
print("There are {} sentences in the corpus.".format(len(data)))
print("There are {} sentences in the training set.".format(len(data.training_set)))
print("There are {} sentences in the testing set.".format(len(data.testing_set)))

""" Have a peek of data.sentences"""

key = 'b100-38532'
print("Sentence: {}".format(key))
print("words:\n\t{!s}".format(data.sentences[key].words))
print("tags:\n\t{!s}".format(data.sentences[key].tags))

"""Counting unique elements in the corpus."""

print("There are a total of {} samples of {} unique words in the corpus."
      .format(data.N, len(data.vocab)))
print("There are {} samples of {} unique words in the training set."
      .format(data.training_set.N, len(data.training_set.vocab)))
print("There are {} samples of {} unique words in the testing set."
      .format(data.testing_set.N, len(data.testing_set.vocab)))
print("There are {} words in the test set that are missing in the training set."
      .format(len(data.testing_set.vocab - data.training_set.vocab)))

"""Accessing words with Dataset.X and tags with Dataset.Y"""

for i in range(2):    
    print("Sentence {}:".format(i + 1), data.X[i])
    print()
    print("Labels {}:".format(i + 1), data.Y[i])
    print()

"""See how data.stream() works."""

print("\nStream (word, tag) pairs:\n")
for i, pair in enumerate(data.stream()):
    print("\t", pair)
    if i > 3: break

"""Pair counts"""

def pair_counts(tags, words):
    d = defaultdict(lambda: defaultdict(int))
    for tag, word in zip(tags, words):
        d[tag][word] += 1
        
    return d
tags = [tag for i, (word, tag) in enumerate(data.training_set.stream())]
words = [word for i, (word, tag) in enumerate(data.training_set.stream())]

def unigram_counts(sequences):
 return Counter(sequences)

tags = [tag for i, (word, tag) in enumerate(data.training_set.stream())]
tag_unigrams = unigram_counts(tags)

def bigram_counts(sequences):
  d = Counter(sequences)
  return d

tags = [tag for i, (word, tag) in enumerate(data.stream())]
o = [(tags[i],tags[i+1]) for i in range(0,len(tags)-2,2)]
tag_bigrams = bigram_counts(o)

def starting_counts(sequences):
    
    d = Counter(sequences)
    return d
tags = [tag for i, (word, tag) in enumerate(data.stream())]
starts_tag = [i[0] for i in data.Y]
tag_starts = starting_counts(starts_tag)

def ending_counts(sequences):
    
    d = Counter(sequences)
    return d
end_tag = [i[len(i)-1] for i in data.Y]
tag_ends = ending_counts(end_tag)

"""Making predictions with a model."""

def replace_unknown(sequence):
    
    return [w if w in data.training_set.vocab else 'nan' for w in sequence]
def simplify_decoding(X, model):
    
    _, state_path = model.viterbi(replace_unknown(X))
    return [state[1].name for state in state_path[1:-1]]

"""Evaluating model accuracy."""

def accuracy(X, Y, model):
    
    correct = total_predictions = 0
    for observations, actual_tags in zip(X, Y):
        
        # The model.viterbi call in simplify_decoding will return None if the HMM
        # raises an error (for example, if a test sentence contains a word that
        # is out of vocabulary for the training set). Any exception counts the
        # full sentence as an error (which makes this a conservative estimate).
        try:
            most_likely_tags = simplify_decoding(observations, model)
            correct += sum(p == t for p, t in zip(most_likely_tags, actual_tags))
        except:
            pass
        total_predictions += len(observations)
    return correct / total_predictions

basic_model = HiddenMarkovModel(name="base-hmm-tagger")
tags = [tag for i, (word, tag) in enumerate(data.stream())]
words = [word for i, (word, tag) in enumerate(data.stream())]
tags_count=unigram_counts(tags)
tag_words_count=pair_counts(tags,words)
starting_tag_list=[i[0] for i in data.Y]
ending_tag_list=[i[-1] for i in data.Y]
starting_tag_count=starting_counts(starting_tag_list)#the number of times a tag occured at the start
ending_tag_count=ending_counts(ending_tag_list)      #the number of times a tag occured at the end

to_pass_states = []
for tag, words_dict in tag_words_count.items():
    total = float(sum(words_dict.values()))
    distribution = {word: count/total for word, count in words_dict.items()}
    tag_emissions = DiscreteDistribution(distribution)
    tag_state = State(tag_emissions, name=tag)
    to_pass_states.append(tag_state)

basic_model.add_states()
start_prob={}
for tag in tags:
    start_prob[tag]=starting_tag_count[tag]/tags_count[tag]
for tag_state in to_pass_states :
    basic_model.add_transition(basic_model.start,tag_state,start_prob[tag_state.name])
end_prob={}
for tag in tags:
    end_prob[tag]=ending_tag_count[tag]/tags_count[tag]
for tag_state in to_pass_states :
    basic_model.add_transition(tag_state,basic_model.end,end_prob[tag_state.name])
transition_prob_pair={}
for key in tag_bigrams.keys():
    transition_prob_pair[key]=tag_bigrams.get(key)/tags_count[key[0]]
for tag_state in to_pass_states :
    for next_tag_state in to_pass_states :
        basic_model.add_transition(tag_state,next_tag_state,transition_prob_pair[(tag_state.name,next_tag_state.name)])
basic_model.bake()

"""Example Decoding Sequences with HMM Tagger"""

for key in data.testing_set.keys[:2]:
    print("Sentence Key: {}\n".format(key))
    print("Predicted labels:\n-----------------")
    print(simplify_decoding(data.sentences[key].words, basic_model))
    print()
    print("Actual labels:\n--------------")
    print(data.sentences[key].tags)
    print("\n")

"""Evaluate the accuracy of the HMM tagger"""

hmm_training_acc = accuracy(data.training_set.X, data.training_set.Y, basic_model)
print("training accuracy basic hmm model: {:.2f}%".format(100 * hmm_training_acc))
hmm_testing_acc = accuracy(data.testing_set.X, data.testing_set.Y, basic_model)
print("testing accuracy basic hmm model: {:.2f}%".format(100 * hmm_testing_acc))

"""MFC tagger"""

FakeState = namedtuple('FakeState', 'name')
class MFCTagger:
    missing = FakeState(name = '<MISSING>')
    
    def __init__(self, table):
        self.table = defaultdict(lambda: MFCTagger.missing)
        self.table.update({word: FakeState(name=tag) for word, tag in table.items()})
        
    def viterbi(self, seq):
        """This method simplifies predictions by matching the Pomegranate viterbi() interface"""
        return 0., list(enumerate(["<start>"] + [self.table[w] for w in seq] + ["<end>"]))
    
tags = [tag for i, (word, tag) in enumerate(data.training_set.stream())]
words = [word for i, (word, tag) in enumerate(data.training_set.stream())]
word_counts = pair_counts(words, tags)
mfc_table = dict((word, max(tags.keys(), key=lambda key: tags[key])) for word, tags in word_counts.items())
mfc_model = MFCTagger(mfc_table)

"""Example Decoding Sequences with MFC Tagger"""

for key in data.testing_set.keys[:2]:
    print("Sentence Key: {}\n".format(key))
    print("Predicted labels:\n-----------------")
    print(simplify_decoding(data.sentences[key].words, mfc_model))
    print()
    print("Actual labels:\n--------------")
    print(data.sentences[key].tags)
    print("\n")

"""Evaluate the accuracy of the MFC tagger"""

mfc_training_acc = accuracy(data.training_set.X, data.training_set.Y, mfc_model)
print("training accuracy mfc_model: {:.2f}%".format(100 * mfc_training_acc))
mfc_testing_acc = accuracy(data.testing_set.X, data.testing_set.Y, mfc_model)
print("testing accuracy mfc_model: {:.2f}%".format(100 * mfc_testing_acc))
