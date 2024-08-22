import json
import itertools
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import metrics
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import argparse

def generate_unique_string(method, **kwargs):
    # Sort the kwargs to ensure consistent ordering
    sorted_params = sorted(kwargs.items())

    # Create a string representation of the parameters
    params_str = '|'.join([f"{key}={value}" for key, value in sorted_params])

    # Concatenate the method name and the parameters string
    unique_string = f"{method}|{params_str}"

    return unique_string

def run_experiment(method, tokenizer, model, device, **kwargs):
    """
    Function to run the experiment with the specified method and parameters.
    """
    length = kwargs['length']

    # move this when the datasize is large
    dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{length}")

    metric = metrics.metric(method, tokenizer, model, device, **kwargs)
    measure_function = metric.get_function()

    measures = []

    for idx, text in enumerate(dataset['input'], 1):
        measures.append(measure_function(text))

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(dataset['input'])} texts.")

    measures = torch.tensor(measures)
    print(f"Total input texts processed: {len(measures)}")

    labels = torch.tensor(dataset['label'])
    indices_0 = torch.where(labels == 0)[0]
    indices_1 = torch.where(labels == 1)[0]

    filtered_uncertainties_0 = measures[indices_0].numpy()
    filtered_uncertainties_1 = measures[indices_1].numpy()

    setting = generate_unique_string(method, **kwargs)
    print(setting)

    filename_base = './ablation/' + setting

    np.save(filename_base + '_unseen.npy', filtered_uncertainties_0)
    np.save(filename_base + '_seen.npy', filtered_uncertainties_1)
    
    measures = measures.numpy()
    # print(measures)
    auc = roc_auc_score(labels, -measures)
    print(f"AUC:{auc}")

    # Plotting histograms
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_uncertainties_0, bins=50, alpha=0.75, color='blue', label='unseen', histtype='step', linewidth=2)
    plt.hist(filtered_uncertainties_1, bins=50, alpha=0.75, color='red', label='seen', histtype='step', linewidth=2)
    plt.title('Histogram by Label')
    plt.xlabel(f"AUC:{auc}")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    plt.savefig(filename_base + '.png')

def main(config_path, device_id):
    with open(config_path, 'r') as file:
        config_methods = json.load(file)

    device = f'cuda:{device_id}' if device_id >= 0 else 'cpu'
    model_name = "huggyllama/llama-13b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.float16)
    model.eval()
    

    for method, config_options in config_methods.items():
        keys, values = zip(*config_options.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for experiment in experiments:
            print(f"Running {method} with configuration: {experiment}")
            run_experiment(method, tokenizer, model, device, **experiment)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process input parameters for the text perturbation and uncertainty measurement script.')
    parser.add_argument('--device', type=int, default=-1, help='GPU device ID to run the experiment on (e.g., 0, 1, 2, 3 for GPUs, or -1 for CPU).')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file for the experiment.')
    args = parser.parse_args()
    
    
    # Specify the GPU device
    device = args.device

    # config_path = "/path/to/your/experiment_config.json"  # Adjust this path to your configuration file
    config_path  = args.config_path

    main(config_path, device)