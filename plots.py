# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_auc_score
# from uncertainty_measurement import measure_entropy, measure_uncertainty_min


# def run_ppl():
#     measures = []
#     for idx, text in enumerate(dataset['input'], 1):
#         measures.append(measure_uncertainty_min(text))
#         if idx % 100 == 0:
#             print(f"Processed {idx}/{len(dataset['input'])} texts.")

#     measures = torch.tensor(measures)
#     print(f"Total input texts processed: {len(measures)}")

#     labels = torch.tensor(dataset['label'])
#     indices_0 = torch.where(labels == 0)[0]
#     indices_1 = torch.where(labels == 1)[0]

#     filtered_uncertainties_0 = measures[indices_0].numpy()
#     filtered_uncertainties_1 = measures[indices_1].numpy()
#     filename_base = f"add{args.add_prob}_perturb{args.perturbations}_kfrac{args.k_fraction}_len{args.length}_minK_"

#     np.save(filename_base+'unseen.npy', filtered_uncertainties_0)
#     np.save(filename_base+'seen.npy', filtered_uncertainties_1)

#     auc = roc_auc_score(labels, -uncertainty_measures)
#     print(f"AUC_ppl:{auc}")


