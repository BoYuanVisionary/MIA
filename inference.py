import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import pertubation

# Setup command-line argument parsing
parser = argparse.ArgumentParser(description='Process input parameters for the text perturbation and uncertainty measurement script.')
# available model names: "huggyllama/llama-7b"
parser.add_argument('--model', type=str, default="huggyllama/llama-7b", help='HuggingFace Model used for inference')
parser.add_argument('--add_prob', type=float, default=0.8, help='Add probability for text perturbation.')
parser.add_argument('--perturbations', type=int, default=10, help='Number of perturbations for uncertainty measurement.')
parser.add_argument('--k_fraction', type=float, default=0.2,help='Fraction of k for minimum uncertainty measurement.')
parser.add_argument('--length', type=int, default=32,help='Length of test sentences in the dataset. Choose one from \{32,64,128,256\}')
# parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save the output files.')
args = parser.parse_args()


def perturb_text_add(text, add_prob=args.add_prob):
    words = text.split()
    perturbed_words = []
    for word in words:
        perturbed_words.append(word)
        # With a certain probability, add a word
        if np.random.rand() < add_prob:
            # Example strategy: repeat the current word
            perturbed_words.append(word)
    
    # Optionally, print the original and perturbed text with a very low probability to check the function's effect
    if np.random.rand() < 0.01:
        print("Original text:", text)
        print("Perturbed text:", ' '.join(perturbed_words))
        
    return ' '.join(perturbed_words)

def measure_uncertainty(input_text, perturbations=args.perturbations):
    perturbed_texts = [pertubation.perturb_token_keywords(input_text) for _ in range(perturbations)] + [input_text]
    probabilities = []

    for pt in perturbed_texts:
        input_ids = tokenizer.encode(pt, return_tensors="pt").to(first_param_device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            log_likelihood = outputs.loss
            probabilities.append((log_likelihood).item() / input_ids.shape[1])

    probabilities = torch.exp(torch.tensor(probabilities))
    uncertainty = torch.log(torch.sum(torch.abs(probabilities - probabilities[-1])) / perturbations)

    return uncertainty.item()

def measure_uncertainty_min(input_text, k_fraction=args.k_fraction):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(first_param_device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        input_ids_processed = input_ids[0][1:]
        all_prob = []
        for i, token_id in enumerate(input_ids_processed):
            prob_token = probs[0, i, token_id].item()
            all_prob.append(prob_token)
        k = int(probs.shape[1] * k_fraction)
        # consider why without all_prob (a wrong implmentation one can also get reasonable results.)
        values, indices = torch.topk(torch.tensor(all_prob), k, largest=False)
        product = torch.mean(-torch.log(values))
    return product


def measure_entropy(input_text, tokenizer, model):
    # Encode the input text and send it to the same device as the model
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    
    # Perform inference without computing gradients
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits
        # Calculate probabilities from logits for each token position
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Compute entropy for the probabilities at each position
        # Entropy formula: H = -sum(p(x) * log(p(x))), where the sum is over all possible tokens
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # Add a small number to avoid log(0)
        
        # Sum entropies across all positions to get the total entropy of the sequence
        mean_entropy = torch.mean(entropy).item()  # Convert to Python scalar

    return mean_entropy

def run_ent_plot():
    uncertainty_measures = []
    for idx, text in enumerate(dataset['input'], 1):
        uncertainty_measures.append(measure_entropy(text))
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(dataset['input'])} texts.")

    uncertainty_measures = torch.tensor(uncertainty_measures)
    print(f"Total input texts processed: {len(uncertainty_measures)}")

    labels = torch.tensor(dataset['label'])
    indices_0 = torch.where(labels == 0)[0]
    indices_1 = torch.where(labels == 1)[0]

    filtered_uncertainties_0 = uncertainty_measures[indices_0].numpy()
    filtered_uncertainties_1 = uncertainty_measures[indices_1].numpy()
    filename_base = f"add{args.add_prob}_perturb{args.perturbations}_kfrac{args.k_fraction}_len{args.length}_ent_"

    np.save(filename_base+'unseen.npy', filtered_uncertainties_0)
    np.save(filename_base+'seen.npy', filtered_uncertainties_1)

    auc = roc_auc_score(labels, -uncertainty_measures)
    print(f"AUC_ent:{auc}")

    # Plotting histograms
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_uncertainties_0, bins=50, alpha=0.75, color='blue', label='unseen', histtype='step', linewidth=2)
    plt.hist(filtered_uncertainties_1, bins=50, alpha=0.75, color='red', label='seen', histtype='step', linewidth=2)

    plt.title('Histogram of Ent by Label')
    plt.xlabel(f"AUC_ent:{auc}")
    plt.ylabel(args.model)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    plt.savefig(filename_base+'.png')  


def run_ppl_plot():
    uncertainty_measures = []
    for idx, text in enumerate(dataset['input'], 1):
        uncertainty_measures.append(measure_uncertainty_min(text))
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(dataset['input'])} texts.")

    uncertainty_measures = torch.tensor(uncertainty_measures)
    print(f"Total input texts processed: {len(uncertainty_measures)}")

    labels = torch.tensor(dataset['label'])
    indices_0 = torch.where(labels == 0)[0]
    indices_1 = torch.where(labels == 1)[0]

    filtered_uncertainties_0 = uncertainty_measures[indices_0].numpy()
    filtered_uncertainties_1 = uncertainty_measures[indices_1].numpy()
    filename_base = f"add{args.add_prob}_perturb{args.perturbations}_kfrac{args.k_fraction}_len{args.length}_minK_"

    np.save(filename_base+'unseen.npy', filtered_uncertainties_0)
    np.save(filename_base+'seen.npy', filtered_uncertainties_1)

    auc = roc_auc_score(labels, -uncertainty_measures)
    print(f"AUC_ppl:{auc}")

    # Plotting histograms
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_uncertainties_0, bins=50, alpha=0.75, color='blue', label='unseen', histtype='step', linewidth=2)
    plt.hist(filtered_uncertainties_1, bins=50, alpha=0.75, color='red', label='seen', histtype='step', linewidth=2)

    plt.title('Histogram of PPL by Label')
    plt.xlabel(f"AUC_ppl:{auc}")
    plt.ylabel(args.model)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    plt.savefig(filename_base+'.png')  

def run_uncertainty_plot():
    uncertainty_measures = []
    for idx, text in enumerate(dataset['input'], 1):
        uncertainty_measures.append(measure_uncertainty(text))
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(dataset['input'])} texts.")

    uncertainty_measures = torch.tensor(uncertainty_measures)
    print(f"Total input texts processed: {len(uncertainty_measures)}")

    labels = torch.tensor(dataset['label'])
    indices_0 = torch.where(labels == 0)[0]
    indices_1 = torch.where(labels == 1)[0]

    filtered_uncertainties_0 = uncertainty_measures[indices_0].numpy()
    filtered_uncertainties_1 = uncertainty_measures[indices_1].numpy()
    filename_base = f"add{args.add_prob}_perturb{args.perturbations}_kfrac{args.k_fraction}_len{args.length}_uncertainty_"

    np.save(filename_base+'unseen.npy', filtered_uncertainties_0)
    np.save(filename_base+'seen.npy', filtered_uncertainties_1)

    auc = roc_auc_score(labels, -uncertainty_measures)
    print(f"AUC_uncertainty:{auc}")

    # Plotting histograms
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_uncertainties_0, bins=50, alpha=0.75, color='blue', label='unseen', histtype='step', linewidth=2)
    plt.hist(filtered_uncertainties_1, bins=50, alpha=0.75, color='red', label='seen', histtype='step', linewidth=2)

    plt.title('Histogram of Uncertainties by Label')
    plt.xlabel(f"AUC:{auc}")
    plt.ylabel(args.model)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    plt.savefig(filename_base+'.png')  

# Main processing loop
if __name__ == "__main__":


    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', torch_dtype=torch.float16)
    model.eval()
    first_param_device = next(model.parameters()).device

    LENGTH = args.length
    dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{LENGTH}")
    # run_uncertainty_plot()
    run_ppl_plot()
    run_uncertainty_plot()


