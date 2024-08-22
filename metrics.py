import torch
import numpy as np
import perturbation
import utils

# def metric_uncertainty(input_text, perturbations, add_prob, delete_prob, model_parameters):

#     model = model_parameters["model"]
#     tokenizer = model_parameters["tokenizer"]
#     first_param_device = model_parameters["first_param_device"]

#     def apply_perturbation(text, add_prob=None, delete_prob=None):
#         if add_prob is not None:
#             return perturbation.perturb_text_add(text, add_prob)
#         elif delete_prob is not None:
#             return perturbation.perturb_text_delete(text, delete_prob)
#         elif (add_prob is None) and (delete_prob is None):
#         utils.

#             return perturbation.perturb_text_adaptive(text)
#         else:
#             raise ValueError("Can not provide both add_prob and delete_prob.")


#     perturbed_texts = [apply_perturbation(input_text, add_prob, delete_prob) for _ in range(perturbations)] + [input_text]
#     probabilities = []

#     for pt in perturbed_texts:
#         input_ids = tokenizer.encode(pt, return_tensors="pt").to(first_param_device)
#         with torch.no_grad():
#             outputs = model(input_ids, labels=input_ids)
#             log_likelihood = outputs.loss
#             probabilities.append((log_likelihood).item() / input_ids.shape[1])

#     probabilities = torch.exp(torch.tensor(probabilities))
#     uncertainty = torch.log(torch.sum(torch.abs(probabilities - probabilities[-1])) / perturbations).item()
#     return uncertainty

# def metric_ppl(input_text, k_fraction, model_parameters):

#     model = model_parameters["model"]
#     tokenizer = model_parameters["tokenizer"]
#     first_param_device = model_parameters["first_param_device"]

#     input_ids = tokenizer.encode(input_text, return_tensors="pt").to(first_param_device)
#     probs = utils.prob(input_ids, model)
#     k = int(probs.shape[1] * k_fraction)
#     # consider why without all_prob (a wrong implmentation one can also get reasonable results.)
#     values, _ = torch.topk(probs, k, largest=False)
#     ppl = torch.exp(torch.mean(-torch.log(values))).item()
#     return ppl


# def measure_entropy(input_text, k_fraction, model_parameters):

#     model = model_parameters["model"]
#     tokenizer = model_parameters["tokenizer"]
#     first_param_device = model_parameters["first_param_device"]

#     input_ids = tokenizer.encode(input_text, return_tensors="pt").to(first_param_device) 
#     entropy = utils.entropy(input_ids, model)
#     k = int(entropy.shape[1] * k_fraction)
#     values, _ = torch.topk(entropy, k, largest=False)
#     mean_entropy = torch.mean(values).item()
#     return mean_entropy

class metric:
    
    def __init__(self,method, tokenizer, model, first_param_device, **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.model = model
        self.tokenizer = tokenizer
        self.first_param_device = first_param_device
    
    def get_function(self):
        
        functions = {
        'mink_ppl': self.mink_ppl,
        'maxk_ppl': self.maxk_ppl,
        'perturb':  self.perturb,
        'mink_entropy': self.mink_entropy,
        'maxk_entropy': self.maxk_entropy,
        'mink_keyword_entropy': self.mink_keyword_entropy,
        'maxk_keyword_entropy': self.maxk_keyword_entropy,
        'weight_gradient': self.weight_gradient,
        'mink_plus': self.mink_plus,
        'mink_score_near': self.mink_score_near
        }
        if self.method in functions.keys():
            if self.method == 'weight_gradient':
                for param in self.model.parameters():
                    param.requires_grad = True    
            return functions[self.method]
        else:
            raise ValueError(f"Method '{self.method}' is not supported.")

        
    def mink_ppl(self, input_text):

        k_fraction = self.kwargs.get('k_fraction', None)
        
        model = self.model
        tokenizer = self.tokenizer
        first_param_device = self.first_param_device

        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(first_param_device)
        probs = utils.prob(input_ids, model)
        k = int(probs.shape[0] * k_fraction)
        values, _ = torch.topk(probs, k, largest=False)
        ppl = torch.exp(torch.mean(-torch.log(values))).item()
        return ppl
    
    def maxk_ppl(self, input_text):

        k_fraction = self.kwargs.get('k_fraction', None)
        
        model = self.model
        tokenizer = self.tokenizer
        first_param_device = self.first_param_device

        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(first_param_device)
        probs = utils.prob(input_ids, model)
        k = int(probs.shape[0] * k_fraction)
        values, _ = torch.topk(probs, k, largest=True)
        ppl = torch.exp(torch.mean(-torch.log(values))).item()
        return ppl
    
    def mink_entropy(self, input_text):

        k_fraction = self.kwargs.get('k_fraction', None)
        
        model = self.model
        tokenizer = self.tokenizer
        first_param_device = self.first_param_device

        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(first_param_device)
        entropys = utils.entropy(input_ids, model)
        k = int(entropys.shape[0] * k_fraction)
        values, _ = torch.topk(entropys, k, largest=False)
        ppl = torch.mean(values).item()
        return ppl
    
    def maxk_entropy(self, input_text):

        k_fraction = self.kwargs.get('k_fraction', None)
        
        model = self.model
        tokenizer = self.tokenizer
        first_param_device = self.first_param_device

        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(first_param_device)
        entropys = utils.entropy(input_ids, model)
        k = int(entropys.shape[0] * k_fraction)
        values, _ = torch.topk(entropys, k, largest=True)
        ppl = torch.mean(values).item()
        return ppl
    
    def weight_gradient(self, input_text):

        k_fraction = self.kwargs.get('k_fraction', None)
        
        model = self.model
        tokenizer = self.tokenizer
        first_param_device = self.first_param_device

        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(first_param_device)
        model.zero_grad()
        gradients = utils.gradient(input_ids, model)
        values, _ = torch.topk(gradients, int(k_fraction*gradients.shape[0]), largest=False)
        small_gradient= torch.sum(values).item()
        return small_gradient
    
    # to do: add keyword_entropy
    def mink_keyword_entropy(self, input_text):
        raise NotImplementedError
        # to do: add keyword_entropy
    def maxk_keyword_entropy(self, input_text):
        raise NotImplementedError
    
    def mink_plus(self, input_text):
        k_fraction = self.kwargs.get('k_fraction', None)
        ratio_entropy = self.kwargs.get('ratio_entropy', None)
        
        model = self.model
        tokenizer = self.tokenizer
        first_param_device = self.first_param_device

        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(first_param_device)
        entropys = utils.entropy(input_ids, model)
        probs = utils.prob(input_ids, model).to(first_param_device)
        scores = torch.log(probs) - ratio_entropy* entropys
        k = int(scores.shape[0] * k_fraction)
        values, _ = torch.topk(scores, k, largest=False)
        score = torch.mean(values).item()
        return -score
    
    def mink_score_near(self, input_text):
        k_fraction = self.kwargs.get('k_fraction', None)
        ratio_entropy = self.kwargs.get('ratio_entropy', None)
        n = self.kwargs.get('n', None)
        
        model = self.model
        tokenizer = self.tokenizer
        first_param_device = self.first_param_device

        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(first_param_device)
        scores = utils.score_near(input_ids, model, n)
        k = int(scores.shape[0] * k_fraction)
        values, _ = torch.topk(scores, k, largest=False)
        score = torch.mean(values).item()
        return -score
        
    
    # There are several other components: fraction and adaptive, and we can also change the keywords, or change it with given probability 
    def perturb(self, input_text):

        model = self.model
        tokenizer = self.tokenizer
        first_param_device = self.first_param_device
        prob = self.kwargs.get('prob', None)
        perturbations = self.kwargs.get('perturbations', None)
        
        target = self.kwargs.get('target', None)
        direction = self.kwargs.get('direction', None)
        if target == 'all' and direction == 'add':
            apply_perturbation = perturbation.perturb_text_add
        elif target == 'all' and direction == 'delete':
            apply_perturbation = perturbation.perturb_text_delete
        elif target == 'keyword' and direction == 'add':
            apply_perturbation = perturbation.perturb_token_keywords_add
        elif target == 'keyword' and direction == 'delete':
            apply_perturbation = perturbation.perturb_token_keywords_delete
        else:
            raise ValueError(f"Target '{target}' and direction '{direction}' is not supported.")
        
        # text level perturbation
        perturbed_texts = [apply_perturbation(input_text, prob) for _ in range(perturbations)] + [input_text]
        probabilities = []

        for pt in perturbed_texts:
            input_ids = tokenizer.encode(pt, return_tensors="pt").to(first_param_device)
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                log_likelihood = outputs.loss
                probabilities.append((log_likelihood).item() / input_ids.shape[1])

        probabilities = torch.exp(torch.tensor(probabilities))
        uncertainty = torch.log(torch.sum(torch.abs(probabilities - probabilities[-1])) / perturbations).item()
        return uncertainty
        
        




