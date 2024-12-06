from utils_tokenize import *
from utils_generation import get_completions
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_activations(prompts, model, nn_model, tokenizer, length = 16, batch_size = 8, file_name = None):
    if file_name is not None and os.path.exists(f"tmp/{file_name}_activations.pt"):
        return torch.load(f"tmp/{file_name}_activations.pt", map_location="cuda:1", weights_only=True)

    prompt_len = arr_tokenize(prompts, tokenizer).shape[1]
    completions = get_completions(prompts, model, tokenizer, length, batch_size, f"{file_name}")

    dataloader = DataLoader(completions, batch_size=batch_size, shuffle=False)

    # Clear initial CUDA cache
    torch.cuda.empty_cache()
    print(f"Initial CUDA memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Initial CUDA memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

    all_layers = torch.zeros(len(completions), 64, length, 4096).to("cuda:1")

    # Process batches
    num_processed = 0


    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Clear CUDA cache before each batch
        torch.cuda.empty_cache()

        new_layers = [None] * 64

        with nn_model.trace(batch) as tracer:
            for index, layer in enumerate(nn_model.backbone.layers):
                new_layers[index] = layer.output[None, :].save()
        
        all_layers[num_processed:num_processed+batch.shape[0]] = torch.cat(new_layers, dim=0).transpose(0, 1)[:, :, prompt_len:].to("cuda:1")
        num_processed += batch.shape[0]

    print(f"activations.device for {file_name}:", all_layers.device)
    print(f"activations.shape for {file_name}:", all_layers.shape)

    if file_name is not None:
        torch.save(all_layers, f"tmp/{file_name}_activations.pt")
    return all_layers