import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cuda")

# Load pretrained GPT-2 (smallest version — 117M params)
print("Loading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # add this line

model     = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.eval()

# Inspect the model
params = sum(p.numel() for p in model.parameters())
print(f"Model loaded!")
print(f"Parameters: {params:,}")
print(f"Device:     {next(model.parameters()).device}")
print(f"\nModel architecture:")
print(model)

# Generate text with GPT-2

def generate(prompt, max_new_tokens=50, temperature=0.8):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n--- GPT-2 Generation ---")
print(generate("The cat sat on the mat and"))
print()
print(generate("Once upon a time in a land far away"))
print()
print(generate("The best way to learn machine learning is"))