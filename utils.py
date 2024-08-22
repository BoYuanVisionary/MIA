# Make sure model and token are on the same device

import torch
from rake_nltk import Rake
import numpy as np
import re
import warnings

# Compute the probability of each token Return: 1 by (len(input_ids)-1) tensor
def prob(input_ids, model):
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # The first token of LLama models is always <s>. remove <s>
        input_ids_processed = input_ids[0][1:]
        all_prob = []
        for i, token_id in enumerate(input_ids_processed):
            prob_token = probs[0, i, token_id]
            all_prob.append(prob_token)
    return torch.tensor(all_prob)

def score_near(input_ids, model, k):
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # The first token of LLama models is always <s>. remove <s>
        input_ids_processed = input_ids[0][1:]
        all_prob = []
        # Tensor to store the result
        top_k_probs = torch.zeros((probs.size(1)-1, k))
    
        for i, token_id in enumerate(input_ids_processed):
            ref_prob = probs[0, i, token_id]
            all_prob.append(torch.log(ref_prob))
            # Filter probabilities that are less than and greater than the reference probability
            lower_mask = probs[0, i, :] <= ref_prob
            higher_mask = probs[0, i,:] > ref_prob
            lower_probs = probs[0, i, lower_mask]
            higher_probs = probs[0, i, higher_mask]
        
            # Sort the filtered probabilities
            sorted_lower, _ = torch.sort(lower_probs, descending=True)
            sorted_higher, _ = torch.sort(higher_probs, descending=True)
        
            # Fill top_k_probs tensor
            if sorted_lower.numel() < k:
                warnings.warn("Not enough lower probabilities, using higher probabilities to fill.")
                sorted_combined, _ = torch.sort(probs[0, i, :], descending=True)
                top_k_probs[i, :] = sorted_combined[-k:]
            else:
                top_k_probs[i, :] = sorted_lower[:k]
                
        top_k_probs_sum = torch.sum(top_k_probs, dim=-1, keepdim=True)
        top_k_probs_normalized = top_k_probs / top_k_probs_sum
        entropies = -torch.sum(top_k_probs_normalized * torch.log(top_k_probs_normalized + 1e-8), dim=-1)      
        
    return torch.tensor(all_prob) + entropies

# Compute the entropy Return: 1 by (len(input_ids)-1) tensor
def entropy(input_ids, model):
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits
        # Calculate probabilities from logits for each token position
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs[0,0:-1,:]
        # Compute entropy for the probabilities at each position
        # Entropy formula: H = -sum(p(x) * log(p(x))), where the sum is over all possible tokens
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).squeeze()  # Add a small number to avoid log(0)
    return entropy

def gradient(input_ids, model):

    outputs = model(input_ids, labels=input_ids)
    log_likelihood = outputs.loss
    log_likelihood.backward()
    
    l2_norm = []
    for param in model.parameters():
        if param.grad is not None:
            l2_norm.append(torch.norm(param.grad, p=2) ** 2)
    return torch.tensor(l2_norm)

def rake_keyword(sentence):
    r = Rake()
    r.extract_keywords_from_text(sentence)
    return(r.get_ranked_phrases())

def find_word_positions(sentence, word_list):
    # Split the sentence into words
    sentence_without_punctuation = re.sub(r'[^\w\s]', '', sentence)
    sentence_words = sentence_without_punctuation.lower().split()
    word_positions_mask = np.zeros(len(sentence_words))
    for i in range(len(sentence_words)):
        for word in word_list:
            sub_words = word.split()
            positions = []
            start_idx = None
            if sentence_words[i] == sub_words[0] or (sub_words[0] in sentence_words[i]) :
                start_idx = i
                match_count = 1
                if match_count == len(sub_words):
                    end_idx = i + 1
                    word_positions_mask[start_idx:end_idx] = 1
                    break   
                for j in range(i+1, len(sentence_words)):
                    if sentence_words[j] == sub_words[match_count]:
                        match_count += 1
                        if match_count == len(sub_words):
                            end_idx = j + 1
                            word_positions_mask[start_idx:end_idx] = 1
                            break   
    return word_positions_mask


