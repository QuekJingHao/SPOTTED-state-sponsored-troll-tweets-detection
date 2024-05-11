# import modules and dependencies
import numpy as np
import pandas as pd
import nltk
import re
import torch
import demoji

from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

def sentence_vectorizer(tokenized_sentence, model, vector_size):
    
    """
    Converts a sentence of words into a numerical numpy array
    
    Parameters
    ----------
    tokenized_sentence: list 
              Array of tokens generated from tokenizer
              
    model: gensim Word2Vec model
             Trained Word2Vec model
             
    vector_size: int
              Size of each sentence vector
    
    
    Returns
    -------
    numpy array
            Array of size vector_size equal to all summed vectors 
    
    """
    
    # get the vocabulary built from word2vec
    built_vocab = list(model.wv.index_to_key)
    
    row_vector = np.zeros(vector_size)

    # set a counter for the number of words in a sentence
    nword = 0
    for word in tokenized_sentence:
        if word in built_vocab:
            row_vector += model.wv[word]
            nword += 1
        else:
            continue
            
    if nword > 0:
        # return the average vector of each sentence
        row_vector /= nword
    
    return row_vector




# in hindsight coming back to revisit this file, this function is actually not really needed and not optimized
def vec_to_features(df, target, sentence_vectors, vector_size):
    
    """
    Takes a dataframe with a column of sentence vectors and transforms that column into a dataframe of features,
    to be passed to any ML classifiers
    
    
    Parameters
    ----------
    df: pandas_dataframe
              dataframe containing the sentence vectors   
    
    target: str
              name of the column containing the labels of each sentence vector
    
    sentence_vectors: array
              sentence embeddings
    
    vector_size: int
               size of each sentence vector
    
    
    Returns
    -------
    pandas_dataframe
           returns a pandas dataframe of size len(df) by vector_size
    
    """
    
    indx = ['w2v_' + str(i + 1) for i in range(vector_size)]
    
    stacked = df[sentence_vectors][0]
    for i in range(len(df) - 1):
        stacked = np.vstack((stacked, df[sentence_vectors][i + 1]))
  
    df_temp = pd.DataFrame(stacked, columns = indx)
    #print(df_temp)
    
    return pd.concat((df[target], df_temp), axis = 1)




def vec_to_features_Tensor(df, target, sentence_vectors, vector_size):
    
    """
    This function is exactly the same as vec_to_features, but optimized using PyTorch Tensors
    
    Takes a dataframe with a column of sentence vectors and transforms that column into a dataframe of features,
    to be passed to any ML classifiers
    

    Parameters
    ----------
    df: pandas_dataframe
              dataframe containing the sentence vectors   
    
    target: str
              name of the column containing the labels of each sentence vector
    
    sentence_vectors: array
              sentence embeddings
    
    vector_size: int
               size of each sentence vector
    
    
    Returns
    -------
    pandas_dataframe
           returns a pandas dataframe of size len(df) by vector_size
    
    """
    
    
    indx = ['w2v_' + str(i + 1) for i in range(vector_size)]
    
    # now, we convert the very first row into a tensor. We push the tensor to GPU if available
    stacked_tensor = torch.from_numpy(df[sentence_vectors][0])
    if torch.cuda.is_available():
        stacked_tensor = stacked_tensor.to('cuda')
    
    for i in range(len(df) - 1):
        
        subsequent_row = torch.from_numpy(df[sentence_vectors][i + 1])
        
        # push to GPU
        if torch.cuda.is_available():
            subsequent_row = subsequent_row.to('cuda')
        
        stacked_tensor = torch.vstack((stacked_tensor, subsequent_row))
        
    # now, we'll need to convert the huge tensor back to numpy array; before that remember to pull the tensor back to cpu memory first
    stacked = stacked_tensor.cpu().numpy()
        
    df_temp = pd.DataFrame(stacked, columns = indx)
    
    return pd.concat((df[target], df_temp), axis = 1)



def text_processing(text):
    
    """
    This function takes a single sentence as string and performs some text cleaning steps
    
    
    
    """
    
    
    # include - these steps yields the best performance
    text_rtn = demoji.replace(text, repl = '')
    text_rtn = text_rtn.lower().strip()
    text_rtn = re.sub(r'#', '', text_rtn)
    text_rtn = re.sub(r'https://t.co/.+', '', text_rtn)
    
    sentence_tokens = nltk.TweetTokenizer().tokenize(text_rtn)
    
    """
    # Discard the following - they seem to decrease the accuracy of the model
    text_rtn = re.sub(r'@', '', text_rtn)
    text_rtn = re.sub(r'\n', ' ', text_rtn)
    sentence_tokens = word_tokenize(text_rtn)
    text_rtn = re.sub(r'[^A-Za-z0-9]+', ' ', text_rtn).strip()
    """
    
    return sentence_tokens


def text_processing_bert(text):
    
    
    """
    This function takes a single sentence as string and performs some text cleaning steps for BERT model
    
    """
    
    # include - these steps yields the best performance
    text_rtn = demoji.replace(text, repl = '')
    text_rtn = text_rtn.lower()
    text_rtn = re.sub(r'@[A-Za-z0-9]+', '', text_rtn).strip()
    text_rtn = re.sub(r'#', '', text_rtn)
    text_rtn = re.sub(r'https://t.co/.+', '', text_rtn)
    text_rtn = re.sub(r'http://t.co/.+', '', text_rtn)
    
    """
    # Discard the following - they seem to decrease the accuracy of the model
    text_rtn = re.sub(r'\n', ' ', text_rtn)
    sentence_tokens = word_tokenize(text_rtn)
    text_rtn = re.sub(r'[^A-Za-z0-9]+', ' ', text_rtn).strip()
    """
    
    return text_rtn