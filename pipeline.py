import os

available_gpus = [2, 3, 4, 5, 6]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, available_gpus))

from transformers import Mamba2ForCausalLM, AutoTokenizer
import torch

from nnsight import NNsight

torch.set_grad_enabled(False)
from utils import *

model_id = 'mistralai/Mamba-Codestral-7B-v0.1'

tokenizer = AutoTokenizer.from_pretrained(model_id, from_slow=True, legacy=False, cache_dir='./codestral-7b-cache')
tokenizer.pad_token = tokenizer.unk_token

model = Mamba2ForCausalLM.from_pretrained(model_id, cache_dir='./codestral-7b-cache').to("cuda:0")
print(model.device)


# Check GPU memory
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
batch_size = 32 if gpu_memory >= 50 else 4
print(f"GPU memory: {gpu_memory:.1f} GB")
print(f"Using batch size: {batch_size}")

# fix parallelism somehow
nn_model = NNsight(model)

harm_prompts = get_dataset("advbench")[:256]
harmless_prompts = get_dataset("alpaca")[:256]

lens = [len(tokenize(prompt, tokenizer)[0]) for prompt in harm_prompts]
print(f"Number of prompts: {len(lens)}")
print(f"Average length: {sum(lens)/len(lens):.1f} tokens")
print(f"Min length: {min(lens)} tokens")
print(f"Max length: {max(lens)} tokens")

prompt_tok_arr = arr_tokenize(harm_prompts, tokenizer)

if os.path.exists("tmp/refusal_vector.pt") and os.path.exists("tmp/layer.txt"):
    best_refusal_vector = torch.load("tmp/refusal_vector.pt", weights_only=True)
    best_layer = int(open("tmp/layer.txt", "r").read())
else:
    advbench_train_activations = get_activations(harm_prompts, model, nn_model, tokenizer, file_name="advbench_train", batch_size=batch_size)
    harmless_train_activations = get_activations(harmless_prompts, model, nn_model, tokenizer, file_name="harmless_train", batch_size=batch_size)
    normalized_vectors = normalize_refusal_vectors(advbench_train_activations, harmless_train_activations)

    refusal_logits = all_ablation_logits(prompt_tok_arr, normalized_vectors, nn_model, batch_size=128)
    print("advbench refusal_logits.shape:", refusal_logits.shape)
    print("advbench refusal_logits.device:", refusal_logits.device)

    # Calculate size in GB
    # Size = num_elements * bytes_per_element / bytes_per_GB
    elements = refusal_logits.numel()  # number of elements in tensor
    bytes_per_element = refusal_logits.element_size()  # bytes per element (4 for float32)
    bytes_per_GB = 1024**3  # bytes per GB
    size_GB = (elements * bytes_per_element) / bytes_per_GB
    print(f"Refusal logits tensor size: {size_GB:.2f} GB")

    raw_logits = unablated_logits(prompt_tok_arr, nn_model, batch_size=batch_size, file_name="advbench_train")
    unablated_probs = torch.softmax(raw_logits, dim=-1).mean(dim=0)
    unablated_probs.device

    # moving slices manually to avoid OOM
    first_dim_shape = refusal_logits.shape[0]
    first_dim_slice = slice(first_dim_shape//2)
    first_dim_slice_2 = slice(first_dim_shape//2, first_dim_shape)
    avg_probs = torch.softmax(refusal_logits[first_dim_slice], dim=-1).mean(dim=2).to("cuda:3")
    avg_probs_2 = torch.softmax(refusal_logits[first_dim_slice_2], dim=-1).mean(dim=2).to("cuda:3")
    avg_probs = torch.cat([avg_probs, avg_probs_2], dim=0)
    scores = -1 * score_probs(avg_probs)
    print("scores.shape:", scores.shape)

    best_refusal_vector, best_layer = grab_best_refusal_vector(scores, normalized_vectors)

completions_tensor = get_completions(harm_prompts, model, tokenizer, file_name="advbench_train", batch_size=batch_size)
print("advbench completions_tensor.shape:", completions_tensor.shape)

first_unablated_toks = completions_tensor[:, -16]
unique_tokens, counts = torch.unique(first_unablated_toks, return_counts=True)
token_distribution = dict(zip(unique_tokens.cpu().tolist(), counts.cpu().tolist()))
sorted_distribution = dict(sorted(token_distribution.items(), key=lambda x: x[1], reverse=True))

train_ablated = ablated_completions(prompt_tok_arr, best_refusal_vector, best_layer, nn_model, batch_size=batch_size, file_name="advbench_train")
print("advbench_test_ablated.shape:", train_ablated.shape)

length = 16

train_ablated_outputs = from_completion_tensor(train_ablated, tokenizer, length=length, file_name="advbench_train", ablated=True)
train_unablated_outputs = from_completion_tensor(completions_tensor, tokenizer, length=length, file_name="advbench_train")

advbench_val = get_dataset("advbench")[256:320]
harmbench = get_dataset("harmbench_val")[:64]
jailbreak = get_dataset("jailbreakbench")[:64] 
malicious = get_dataset("malicious_instruct")[:64]
strongreject = get_dataset("strongreject")[:64]
tdc2023 = get_dataset("tdc2023")[:64]

names = ["advbench_train", "advbench_test", "harmbench_val", "jailbreak", "malicious", "strongreject", "tdc2023"]
full_names = ["AdvBench Train", "AdvBench Val", "HarmBench", "Jailbreak", "Malicious", "StrongReject", "Tdc2023"]
prompt_dataset = [harm_prompts, advbench_val, harmbench, jailbreak, malicious, strongreject, tdc2023]
prompt_tok_arrs = [arr_tokenize(dataset, tokenizer) for dataset in prompt_dataset]

test_ablated_tensors = []
test_unablated_tensors = []
test_ablated_outputs = []
test_unablated_outputs = []

for index, prompt_tok_arr in enumerate(prompt_tok_arrs):
    file_name = names[index]
    test_ablated_tensors.append(ablated_completions(prompt_tok_arr, best_refusal_vector, best_layer, nn_model, batch_size=batch_size, length=length, file_name=file_name))
    test_unablated_tensors.append(get_completions(prompt_dataset[index], model, tokenizer, batch_size=batch_size, length=length, file_name=file_name))
    test_ablated_outputs.append(from_completion_tensor(test_ablated_tensors[index], tokenizer, length=length, file_name=file_name, ablated=True))
    test_unablated_outputs.append(from_completion_tensor(test_unablated_tensors[index], tokenizer, length=length, file_name=file_name))

first_ablated_toks = torch.cat([test_ablated_tensors[index][:, -16] for index in range(len(test_ablated_tensors))])
first_unablated_toks = torch.cat([test_unablated_tensors[index][:, -16] for index in range(len(test_unablated_tensors))])

print("Number refused for train ablated outputs:", count_refusals(train_ablated_outputs))
print("Number refused for train unablated outputs:", count_refusals(train_unablated_outputs))

for index, (ablated_output, unablated_output) in enumerate(zip(test_ablated_outputs, test_unablated_outputs)):
    print(f"Dataset {names[index]}")
    print("Number refused for ablated outputs:", count_refusals(ablated_output))
    print("Number refused for unablated outputs:", count_refusals(unablated_output))

harmless_test_prompts = get_dataset("alpaca")[256:320]
harmless_test_tok_arr = arr_tokenize(harmless_test_prompts, tokenizer)
harmless_test_completions = get_completions(harmless_test_prompts, model, tokenizer, batch_size=batch_size, length=length, file_name="harmless_test")
harmless_test_activated_completions = get_activated_completions(harmless_test_tok_arr, best_refusal_vector, best_layer, nn_model, batch_size=batch_size, length=length, file_name="harmless_test")
harmless_test_output = from_completion_tensor(harmless_test_completions, tokenizer, length=length, file_name="harmless_test")
harmless_test_activated_output = from_completion_tensor(harmless_test_activated_completions, tokenizer, length=length, file_name="harmless_test_activated")

first_harmless_toks = harmless_test_completions[:, -16]
first_harmless_activated_toks = harmless_test_activated_completions[:, -16]

plot_refusal_scores(full_names, test_ablated_outputs, test_unablated_outputs)
plot_harmless_refusal_scores(harmless_test_output, harmless_test_activated_output)
plot_first_tok_dist(first_ablated_toks, first_unablated_toks, tokenizer, prompt_type="harmful")
plot_first_tok_dist(first_harmless_activated_toks, first_harmless_toks, tokenizer, prompt_type="harmless")

activation_factors = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
scores = []
for activation_factor in activation_factors:
    scores.append(refusal_score_activation_factor(harmless_test_tok_arr, best_refusal_vector, best_layer, tokenizer, nn_model, activation_factor, batch_size=batch_size, file_name="harmless_test"))

plot_line_harmless_refusal_scores(scores, activation_factors)