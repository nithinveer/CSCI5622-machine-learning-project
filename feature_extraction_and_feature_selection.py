#!/usr/bin/env python
# coding: utf-8

# importing all the required libraries
import numpy as np
import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from gensim.models import Word2Vec
import sklearn
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from extract_from_es import getData
import matplotlib.pyplot as plt
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def cleaning_reviews(hotel_id, reviews):
    # cleaning review text by removing
    # punctuations except full stop,
    # normalize into lower cases,
    # remove numeric characters
    # parameters
    # hotel_id: hotel id numbers
    # reviews: review corpus per hotel
    # return: pre-processed reviews
    clean_reviews = {}
    to_delete = set(string.punctuation) - {'.'}
    # remove punctuations except full stop
    string_punc = "".join(to_delete)
    clean_reviews[hotel_id]= {}
    for key in reviews.keys():
        text = str(reviews[key]).split()
        table = str.maketrans('', '', string_punc)
        # cleaning text based on punctuation
        stripped_punctuation = [w.translate(table) for w in text]
        # normalize the letters to lowercase
        normalize_to_lower = [w.lower() for w in stripped_punctuation]
        # remove numeric characters in the text
        clean_words = [x for x in normalize_to_lower if not re.match("[0-9]", x)]
        clean_sent = ' '.join(clean_words)
        clean_reviews[hotel_id][key] = clean_sent
    return clean_reviews 

def reviews_into_sentences(cleaned_reviews):
    # break review sentences by fullstop
    # and store them in dictionary
    # parameters
    # cleaned_reviews: pre-processed review as text
    # return: sentences from each review
    for key in cleaned_reviews.keys():
            sentence_filter = []
            # split the sentences by full stop
            sentences = str(cleaned_reviews[key]).split(".")
            for sentence in sentences:
                if sentence!='':
                    sentence = sentence.strip()
                    sentence_filter.append(sentence)
                    # store the sentences in dictionary
                    cleaned_reviews[key] = sentence_filter
    return cleaned_reviews

# calculate sentiment scores for each sentence
def calculate_sentiment(cleaned_reviews):
   # Calculating the sentiment score per sentence
   # parameters:
   # clean_reviews: cleaned reviews
   # return: sentiment scores per score
   a1 = SentimentIntensityAnalyzer()
   sentiment_scores = {}
   # storing sentimet scores in a dictionary
   for key in cleaned_reviews.keys():
       sentiment_scores[key] = {}
       for sent_index, sentence in enumerate(cleaned_reviews[key]):
           scores = a1.polarity_scores(sentence)
           sentiment_scores[key][sent_index] = scores
   return sentiment_scores

def find_features(cleaned_reviews, sentiment_scores):
    # find features based on POS tagging per sentence
    # and store the features from the sentence in dictionary
    # parameters:
    # cleaned_reviews: pre-processed reviews
    # sentimet_scores: sentiment scores per sentence
    # return: features from each sentence based on POS tagging
    sentences_features = {}
    for key in cleaned_reviews.keys():
        sentences_features[key] = {}
        fet_lst = []
        for sentence_index, sentence in enumerate(cleaned_reviews[key]):
            feature_list = []
            word_tokens = nltk.word_tokenize(sentence)
            result = nltk.pos_tag(word_tokens)
            pos_tag_list, word_list, word_fin_list, pos_tag_fin_list = [],[],[],[]
            for i in range(len(result)):
                pos_tag_list.append(result[i][1])
                word_list.append(result[i][0])
            word_fin_list.append(word_list)
            pos_tag_fin_list.append(pos_tag_list)
            # filtering the features based on POS tagging
            for w in range(len(word_fin_list)):
                for j in range(len(word_fin_list[w])):
                    # Check for nouns
                    if pos_tag_fin_list[w][j]=='NN':
                        for k in range(j+1,len(word_fin_list[w])):
                            # check for conjuction
                            if pos_tag_fin_list[w][k]=='CC':
                                break
                            # Checkf for adjective
                            if pos_tag_fin_list[w][k]=='JJ':
                                feature_list.append(word_fin_list[w][j])
                                break
                        sentences_features[key][sentence_index] = feature_list
                        fet_lst.append(feature_list)
        sentences_features[key]["feature_list"] = fet_lst
    return sentences_features

def get_unique_features(sentences_features):
    # Find unique features and store them in dictionary with key as hotel id
    # parameters:
    # sentences_features: features per sentence
    # return: distinct features filtered per sentence
    unique_features = {}
    for key in sentences_features.keys():
        unique_features[key] = {}
        feat = sentences_features[key]["feature_list"]
        fet_lst = []
        fet_set = set()
        temp= []
        for each_ in feat:
            temp_str = "".join(each_)
            if temp_str not in fet_set:
                temp.append(each_)
                fet_set.add(temp_str)
        unique_features[key]["feature_list"] = temp
    return unique_features

# Transform feature list into embeddiung by using wordtovec
def word_to_vec_hotel(hotel_id, unique_features):
    # Transform features into embeddings using wordtovec module
    # parameters:
    # hotel_id: hotel identification number
    # unique_features: unique features per sentence
    # return: return dictionary with key as hotel id and value as dataframe
    data_month = hotel_features = {}
    hotel_features[hotel_id] = {}
    for key in unique_features.keys():
        md = Word2Vec(unique_features[key]["feature_list"],min_count=1)
        words_ = list(md.wv.vocab)
        word2vec = {}
        for feat_name in unique_features[key]["feature_list"]:
            for f in feat_name:
                if f in md.wv.vocab:
                    word2vec[f] = md[f]
        # converting wod embedding into dataframes
        data_month[key] = pd.DataFrame.from_dict(word2vec)
    hotel_features[hotel_id] = data_month
    return hotel_features


hotel_ids = ["611947","1418811","111428","84217"]
#hotel_ids = ["84217"]
hotel_data = {}
for ids in hotel_ids:   
    reviews = getData(ids)
    # cleaning text
    cleaned_reviews = cleaning_reviews(ids,reviews)
    # break sentences based on full stop
    review_sentences = reviews_into_sentences(cleaned_reviews[ids])
    # Calculate sentiment scores
    sentiment_scores_sentence = calculate_sentiment(review_sentences)
    # find features based on POS tagging for each sentence
    sentences_features = find_features(review_sentences, sentiment_scores_sentence)
    # get unique features and remove redundacy of features.
    unique_sentence_features = get_unique_features(sentences_features)
    # implementing word to vec for each feture list and convert into dataframe.
    hotel_features = word_to_vec_hotel(ids, unique_sentence_features)
    # store bcak each dataframe into key of a dictionary.
    hotel_data[ids] = hotel_features

# identify common features across all the months per hotel.
comm_feat= {}
for i in hotel_ids:
    df_intersect = []
    comm_feat[i]={}
    for j in list(hotel_data[i].keys())[1:len(list(hotel_data[i].keys()))-1]:
        df_keys_1 = list(hotel_data[i][j].keys())
        df_keys_2 = list(hotel_data[i][j].keys())
        df_key_set_1 = set(df_keys_1)
        df_key_set_2 = set(df_keys_2)
        # set intersection to identify common features
        df_intersect.append((df_key_set_1.intersection(df_key_set_2)))   
    comm_feat[i]= df_intersect
    
common_feature_list = {}
for i in hotel_ids:
    common_feature_list[i]= set.intersection(*comm_feat[i])

# storing the common features into one dataframe based on hotelid
df= {}
for i in hotel_ids:
    df[i] = {}
    d1= {}
    dt = getData(i)
    for j in list(hotel_data[i].keys())[1:len(list(hotel_data[i].keys()))-1]:
        d1[j] = {}
        d1[j] = pd.DataFrame(hotel_data[i][j][common_feature_list[i]].sum(axis = 0)).transpose()
    df[i] = pd.concat(d1)
    
#### Feature selection method 1 ##################
################## Implementing Pearson correlation coefficient #####################
# calculate pearson colerration coefficient without PCA
cr = {}
for i in hotel_ids:
    cr[i]= df[i].corr(method='pearson')

#### Feature selection method 2 ##################
################## Implementing Principal component analysis followed by Pearson correlation method #####################

# Using PCA to select important features form each component and followed by
# perason correlation calculation on most important features

data = df[hotel_ids[0]]
# performing standard scalar to normalize the data
scalar = StandardScaler()
# fitting the scalar data
scaled_fit = scalar.fit(data)
scaled_data = scalar.transform(data)
# Implement PCA with target varaince of 99 percent
pca = PCA(n_components=0.99)
# fitting the pca
pca_fit = pca.fit_transform(scaled_data)


# identify the features which have highest loading score per component
n_pca_comp = pca.components_.shape[0]
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pca_comp)]
most_important_features = data.columns
most_important_features = [most_important_features[most_important[i]] for i in range(n_pca_comp)]
data_with_imp_features = data[most_important_features]


# implement perason colerration coefficent among important features and
# returns correlation matrix
data_with_imp_features.corr(method='pearson')





