import numpy as np 
import sys
import os

current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.abspath(os.path.join(current_folder, '..'))
sys.path.append(parent_folder)
from perturbation import perturb_token_keywords_add, perturb_token_keywords_delete
from utils import rake_keyword

# Test Case 1: No addition when add_prob is 0
print("Test Case 1: No addition when add_prob is 0")
text = "Wikipedia is a free online encyclopedia, created and edited by volunteers around the world and hosted by the Wikimedia Foundation."
print(f"key words of the input:{rake_keyword(text)}")
add_prob = 0
result = perturb_token_keywords_add(text, add_prob)
print("Input text:", text)
print("Result:", result)
print()

# Test Case 2: All non-keywords duplicated when add_prob is 1
print("Test Case 2: All non-keywords duplicated when add_prob is 1")
add_prob = 1
result = perturb_token_keywords_add(text, add_prob)
print("Input text:", text)
print("Result:", result)
print()

# Test Case 3: Random behavior with a specific random seed
print("Test Case 3: Random behavior with a specific random seed")
np.random.seed(0)  # Set a seed to make it deterministic
add_prob = 0.5
result = perturb_token_keywords_add(text, add_prob)
print("Input text:", text)
print("Result:", result)
print()

# Test Case 4: Keywords should not be duplicated
print("Test Case 4: All non-keywords should be removed")
add_prob = 1
result = perturb_token_keywords_delete(text, add_prob)
print("Input text:", text)
print("Result:", result)
print()