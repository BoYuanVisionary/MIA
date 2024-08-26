import numpy as np
import re
from utils import rake_keyword

def perturb_text_add(text, add_prob):
    words = text.split()
    perturbed_words = []
    for word in words:
        perturbed_words.append(word)
        if np.random.rand() < add_prob:
            perturbed_words.append(word)
    return ' '.join(perturbed_words)

def perturb_text_delete(text, delete_prob):
    words = text.split()
    perturbed_words = []
    for word in words:
        if np.random.rand() > delete_prob:
            perturbed_words.append(word)
    return ' '.join(perturbed_words)

# to do: adaptive 
# def perturb_token_adaptive(tokens):

#     perturbed_token = []
#     for token in tokens:
#         if np.random.rand() > delete_prob:
#             perturbed_words.append(word)
#     return ' '.join(perturbed_words)

# only perturb non-keyword tokens in the following two ones. 
# note that 1. the keywords are extracted by rake_keyword 2. be careful of text = re.sub(r'[^\w\s]', '', text)
def perturb_token_keywords_add(text, add_prob):
    # add_prob is the probability to remove non-essential words. All the keywords will be fixed. 
    keywords = rake_keyword(text)
    text = re.sub(r'[^\w\s]', '', text) # remove all punctuation
    low_words = text.lower().split()
    words = text.split()
    perturbed_words = []
    index = 0
    while index < len(low_words):
        initial_index = index
        for keyword in keywords:
            start = index
            end = index + len(keyword.split())
            if end <= len(low_words) and ' '.join(low_words[start:end]) == keyword:
                index += len(keyword.split())
                perturbed_words.append(' '.join(words[start:end]))
                break
        if initial_index == index: # meaning this is not the keyword
            perturbed_words.append(words[index])
            if np.random.rand() < add_prob:
                perturbed_words.append(words[index])
            index = index + 1
    return ' '.join(perturbed_words)
    
    
def perturb_token_keywords_delete(text, delete_prob):
    # delete_prob is the probability to remove non-essential words. All the keywords will be fixed. 
    keywords = rake_keyword(text)
    text = re.sub(r'[^\w\s]', '', text) # remove all punctuations
    low_words = text.lower().split()  
    words = text.split()
    perturbed_words = []
    index = 0
    while index < len(low_words):
        initial_index = index
        for keyword in keywords:
            start = index
            end = index + len(keyword.split())
            if end <= len(low_words) and ' '.join(low_words[start:end]) == keyword:
                index += len(keyword.split())
                perturbed_words.append(' '.join(words[start:end]))
                break
        if initial_index == index: # meaning this is not the keyword
            if np.random.rand() > delete_prob:
                perturbed_words.append(words[index])
            index = index + 1
    return ' '.join(perturbed_words)
            
                
                


