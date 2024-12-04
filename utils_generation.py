from utils_tokenize import *
import os
import torch
from torch.utils.data import DataLoader
from einops import einsum
from tqdm import tqdm

def get_completions(prompts, length=16, batch_size=8, file_name=None):
    if file_name is not None and os.path.exists(f"tmp/{file_name}_completions.pt"):
        return torch.load(f"tmp/{file_name}_completions.pt", map_location="cuda:1", weights_only=True)

    prompt_tok_arr = arr_tokenize(prompts)
    prompt_len = prompt_tok_arr.shape[1]
    print(f"Doing inference on {len(prompts)} prompts")

    completions = torch.zeros(prompt_tok_arr.shape[0], prompt_tok_arr.shape[1] + length).to("cuda:1")
    dataloader = DataLoader(prompt_tok_arr, batch_size=batch_size, shuffle=False)
    num_processed = 0
    for batch in tqdm(dataloader, desc="Inference batches"):
        completions[num_processed:num_processed+batch.shape[0]] = infer(batch, length).to("cuda:1")
        num_processed += batch.shape[0]

    completions = completions.to(int)

    if file_name is not None:
        torch.save(completions, f"tmp/{file_name}_completions.pt")

    return completions

def ablated_completions(prompt_toks, refusal_vector, layer_ind, nn_model, batch_size=8, length=16, file_name=None):
    if file_name is not None and os.path.exists(f"tmp/{file_name}_ablated_completions.pt"):
        return torch.load(f"tmp/{file_name}_ablated_completions.pt", map_location="cuda:1", weights_only=True)

    dataloader = DataLoader(prompt_toks, batch_size=batch_size, shuffle=False)
    ablated_completions = torch.zeros(prompt_toks.shape[0], prompt_toks.shape[1] + length).to("cuda:1").to(int)
    num_processed = 0
    for batch in tqdm(dataloader, desc="Processing batches"):
        ablated_completions[num_processed:num_processed+batch.shape[0], :prompt_toks.shape[1]] = batch

        for tok_ind in range(length):
            cur_batch = ablated_completions[num_processed:num_processed+batch.shape[0], :prompt_toks.shape[1]+tok_ind]

            with nn_model.trace(cur_batch):
                l_output_before = nn_model.backbone.layers[layer_ind].output.clone().save()
                dots = einsum(l_output_before, refusal_vector, "b h d, d -> b h")[:, :, None]
                l_output_after = l_output_before - refusal_vector * dots.repeat(1, 1, 4096)
                nn_model.backbone.layers[layer_ind].output = l_output_after
                out = nn_model.output.save()
            new_toks = torch.argmax(out.logits[:, -1], dim=-1)
            ablated_completions[num_processed:num_processed+batch.shape[0], prompt_toks.shape[1]+tok_ind] = new_toks

        num_processed += batch.shape[0]
    
    if file_name is not None:
        torch.save(ablated_completions, f"tmp/{file_name}_ablated_completions.pt")

    return ablated_completions

def get_ablation_logits(prompt_tok_arr, normalized_vectors, batch_size=4):
    if not os.path.exists("tmp/all_logits.pt"):
        cutoff = 0.8
        n_layers = int(0.8 * 64)
        all_logits = torch.zeros(n_layers, normalized_vectors.shape[1], len(prompt_tok_arr), 32768)
        for layer_ind in tqdm(range(n_layers), desc="Processing layers"):
            all_logits[layer_ind] = ablation_logits(prompt_tok_arr, normalized_vectors[layer_ind], layer_ind, batch_size)
        torch.save(all_logits, "tmp/all_logits.pt")
    else:
        all_logits = torch.load("tmp/all_logits.pt", map_location="cuda:2", weights_only=True)
    return all_logits

def ablation_logits(prompt_toks, refusal_vectors, layer_ind, nn_model, batch_size=8):
    # for each refusal vector, generate first-token logits at layer_ind
    # refusal vectors assumed to be normalized
    logits = torch.zeros(len(refusal_vectors), len(prompt_toks), 32768).to("cuda:2")
    dataloader = DataLoader(prompt_toks, batch_size=batch_size, shuffle=False)

    print(f"Processing {len(refusal_vectors)} refusal vectors")
    for vec_ind, vec in enumerate(tqdm(refusal_vectors, desc="Processing refusal vectors", leave=False)):
        num_processed = 0
        for batch_idx, batch in enumerate(dataloader):
            with nn_model.trace(batch):
                l_output_before = nn_model.backbone.layers[layer_ind].output.clone().save()
                dots = einsum(l_output_before, vec, "b h d, d -> b h")[:, :, None]
                l_output_after = l_output_before - vec * dots.repeat(1, 1, 4096)
                nn_model.backbone.layers[layer_ind].output = l_output_after
                out = nn_model.output.save()
            logits[vec_ind, num_processed:num_processed+batch.shape[0]] = out.logits[:, -1].to("cuda:2")
            num_processed += batch.shape[0]
    return logits

def unablated_logits(prompt_toks, nn_model, batch_size=8):
    dataloader = DataLoader(prompt_toks, batch_size=batch_size, shuffle=False)
    logits = torch.zeros(len(prompt_toks), 32768).to("cuda:2")
    num_processed = 0
    for batch in tqdm(dataloader, desc="Processing batches"):
        with nn_model.trace(batch):
            out = nn_model.output.save()
        logits[num_processed:num_processed+batch.shape[0]] = out.logits[:, -1].to("cuda:2")
        num_processed += batch.shape[0]
    return logits

