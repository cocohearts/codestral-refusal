
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

def infer(prompt_toks, model, n_tokens=16):
    # generates all tokens, including new ones
    out = model.generate(prompt_toks, max_new_tokens=n_tokens)
    return out
