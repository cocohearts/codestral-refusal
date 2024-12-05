# Perplexity-generated prompt
import numpy as np
import os
from einops import einsum
import json
import torch
from tqdm import tqdm
from utils_generation import *
from utils_interp import *
from utils_eval import *

def normalize_refusal_vectors(harmful_vec_activations, harmless_vec_activations):
    if not os.path.exists("tmp/normalized_vectors.pt"):
        mean_harmful_activations = harmful_vec_activations.mean(dim=0)
        mean_harmless_activations = harmless_vec_activations.mean(dim=0)
        normalized_vectors = mean_harmful_activations - mean_harmless_activations
        normalized_vectors = normalized_vectors / torch.norm(normalized_vectors, dim=-1, keepdim=True)
        torch.save(normalized_vectors, "tmp/normalized_vectors.pt")
    else:
        normalized_vectors = torch.load("tmp/normalized_vectors.pt", map_location="cuda:1", weights_only=True)
    return normalized_vectors

def score_probs(probs):
    # score correlated with refusal
    scores = probs[..., refusal_tokens].sum(dim=-1)
    return torch.logit(scores)

def fast_refusals(prompt_toks, nn_model, batch_size=8):
    logits = unablated_logits(prompt_toks, nn_model, batch_size)
    probs = torch.softmax(logits, dim=-1)
    return score_probs(probs)

def grab_best_refusal_vector(scores, normalized_vectors, top_n=10):
    if os.path.exists("tmp/refusal_vector.pt") and os.path.exists("tmp/layer.txt"):
        best_refusal_vector = torch.load("tmp/refusal_vector.pt")
        best_layer = int(open("tmp/layer.txt", "r").read())
        return (best_refusal_vector, best_layer)
    flat_indices = torch.argsort(scores.flatten(), descending=True)[:top_n]
    i_indices = flat_indices // scores.shape[1]  # Get row indices
    j_indices = flat_indices % scores.shape[1]   # Get column indices

    print(f"Top {top_n} (layer, token) pairs with highest KL divergence:")
    for idx in range(top_n):
        i, j = i_indices[idx], j_indices[idx]
        print(f"({i.item()}, {j.item()}): {scores[i,j].item():.6f}")
    best_refusal_vector = normalized_vectors[i_indices[0], j_indices[0]]
    best_layer = i_indices[0]
    # Save the best refusal vector and layer
    torch.save(best_refusal_vector, "tmp/refusal_vector.pt")
    with open("tmp/layer.txt", "w") as f:
        f.write(str(best_layer.item()))
    return (best_refusal_vector, best_layer)

def get_dataset(dataset_name):
    with open(f"refusal_direction/dataset/processed/{dataset_name}.json", "r") as f:
        items = json.load(f)
    return [item["instruction"] for item in items]
