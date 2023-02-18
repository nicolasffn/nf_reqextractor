import nltk
from nltk import UnigramTagger
from nltk.corpus import brown
from sklearn import metrics

def process_text(text):
    # tokenize the text into words
    tokens = nltk.word_tokenize(text)

    # perform part-of-speech tagging on the tokens
    tagged_tokens = nltk.pos_tag(tokens)
    # print(tagged_tokens)

    # identify named entities
    entities = tagged_tokens

    # filter out non-noun and non-verb words
    filtered_entities = [(token, tag) for token, tag in entities if tag.startswith("NN") or tag.startswith("VB") or tag.startswith("RB") or tag.startswith("MD") or tag.startswith("WP") or tag.startswith("JJ") or tag.startswith("IN")]

    # we'll use the brown corpus with universal tagset for readability
    tagged_sentences = brown.tagged_sents(categories="news", tagset="universal")
    # let's keep 20% of the data for testing, and 80 for training
    i = int(len(tagged_sentences)*0.2)
    train_sentences = tagged_sentences[i:]
    test_sentences = tagged_sentences[:i]

    # let's train the tagger with out train sentences
    unigram_tagger = UnigramTagger(train_sentences)
    # now let's evaluate with out test sentences
    # default evaluation metric for nltk taggers is accuracy
    accuracy = unigram_tagger.evaluate(test_sentences)
    # print("Accuracy:", accuracy)

    tagged_test_sentences = unigram_tagger.tag_sents([[token for token,tag in sent] for sent in test_sentences])
    gold = [str(tag) for sentence in test_sentences for token,tag in sentence]
    pred = [str(tag) for sentence in tagged_test_sentences for token,tag in sentence]
    
    # print(metrics.classification_report(gold, pred))

    test_sentences = [[word for word in nltk.word_tokenize(text)]]
    tagged_test_sentences = unigram_tagger.tag_sents(test_sentences)
    # print("Accuracy on the given sentence:", tagged_test_sentences)

    return filtered_entities