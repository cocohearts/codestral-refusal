import os
import json
sys_prompt = "As an AI assistant, your core function is to help users while safeguarding against potential misuse. You must refuse any requests that could lead to harm, illegal activities, or the spread of misinformation. When declining, offer a brief explanation and suggest safer alternatives when appropriate."

def wrap(prompt, sys_prompt):
    return "[INST] " + sys_prompt + " [/INST]\n\n" + prompt + "\n\n"

def tokenize(prompt, tokenizer, sys_prompt=sys_prompt):
    return tokenizer(wrap(prompt, sys_prompt), return_tensors="pt")["input_ids"].to("cuda:0")

def arr_tokenize(prompts, tokenizer):
    prompts = [wrap(prompt, sys_prompt) for prompt in prompts]
    return tokenizer(prompts, return_tensors="pt", padding=True)["input_ids"].to("cuda:0")

def generate(prompt, model, tokenizer, n_tokens=16):
    # generates n_tokens text
    tokens = tokenize(prompt, tokenizer)
    out = model.generate(tokens, max_new_tokens=n_tokens)
    new_toks = out[:, tokens.shape[1]:]
    return tokenizer.batch_decode(new_toks)[0]

def from_completion_tensor(completion_tensor, tokenizer, length, file_name=None, ablated=False):
    # returns list of decoded outputs
    if ablated:
        file_name = f"{file_name}_ablated"
    if file_name is not None and os.path.exists(f"tmp/{file_name}_outputs.json"):
        with open(f"tmp/{file_name}_outputs.json", "r") as f:
            return json.load(f)
    outputs = tokenizer.batch_decode(completion_tensor[:, -length:])
    if file_name is not None:
        with open(f"tmp/{file_name}_outputs.json", "w") as f:
            json.dump(outputs, f)
    return outputs