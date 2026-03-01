import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import LoraConfig, get_peft_model

device = torch.device("cuda")

# Load GPT-2
print("Loading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Apply LoRA
lora_config = LoraConfig(
    r=8,                          # rank — how many adapter params
    lora_alpha=32,                # scaling factor
    target_modules=["c_attn"],    # which layers to adapt
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Compare trainable vs frozen params
total    = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters:     {total:,}")
print(f"Trainable parameters: {trainable:,}")
print(f"Trainable %:          {100 * trainable / total:.2f}%")


# Custom training data — teach GPT-2 to answer ML questions
training_data = [
    "Question: What is a neural network? Answer: A neural network is a machine learning model inspired by the brain, made of layers of connected neurons that learn patterns from data.",
    "Question: What is gradient descent? Answer: Gradient descent is an optimization algorithm that minimizes loss by updating weights in the direction that reduces the error.",
    "Question: What is overfitting? Answer: Overfitting is when a model memorizes training data and fails to generalize to new unseen data.",
    "Question: What is a GPU? Answer: A GPU is a graphics processing unit that accelerates machine learning by performing thousands of parallel computations simultaneously.",
    "Question: What is backpropagation? Answer: Backpropagation is the algorithm that computes gradients by propagating the error backwards through the network layers.",
]

def tokenize(text):
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    return tokens["input_ids"].to(device)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("\nFine-tuning on ML Q&A data...")
model.train()
for epoch in range(50):
    total_loss = 0
    for text in training_data:
        input_ids = tokenize(text)
        outputs   = model(input_ids, labels=input_ids)
        loss      = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:2d} | Loss: {total_loss/len(training_data):.4f}")

# Train longer with more examples
for epoch in range(200):    # increase from 50 to 200
    total_loss = 0
    for text in training_data:
        input_ids = tokenize(text)
        outputs   = model(input_ids, labels=input_ids)
        loss      = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 40 == 0:
        print(f"  Epoch {epoch:3d} | Loss: {total_loss/len(training_data):.4f}")

# Generate answers to ML questions
def answer(question, max_new_tokens=60):
    model.eval()
    prompt = f"Question: {question} Answer:"
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
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3    # penalize repeating phrases
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n--- Fine-tuned GPT-2 Answers ---")
print(answer("What is a neural network?"))
print()
print(answer("What is overfitting?"))
print()
print(answer("What is a GPU?"))        
