import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from datasets import load_dataset

# Load model and tokenizer
model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
teacher_model = AutoModelForCausalLM.from_pretrained(model_name)
student_model = AutoModelForCausalLM.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("hans")

# Select the split to use (e.g., 'train')
train_dataset = dataset['train']

def teacher_template(premise, hypothesis):
    return f"Premise: '{premise}'. Hypothesis: '{hypothesis}'. Explain if the premise implies the hypothesis."

def student_template(premise, hypothesis):
    return f"Does '{premise}' imply '{hypothesis}'?"

# Preprocess dataset function
def preprocess_data(item):
    teacher_prompt = teacher_template(item['premise'], item['hypothesis'])
    student_prompt = student_template(item['premise'], item['hypothesis'])
    return {'teacher_prompt': teacher_prompt, 'student_prompt': student_prompt}

processed_dataset = train_dataset.map(preprocess_data)
data_loader = DataLoader(processed_dataset, batch_size=8, shuffle=True)


def kl_divergence_loss(student_logits, teacher_logits):
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return kl_loss

optimizer = AdamW(student_model.parameters(), lr=5e-5)
student_model.train()
teacher_model.eval()

# Training loop
max_length = 512  # Define a max_length that suits your model's capacity

for epoch in range(3):  # Number of epochs
    for batch in data_loader:
        # Tokenize and encode prompts with consistent max_length
        teacher_encoded = tokenizer(batch['teacher_prompt'], return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        student_encoded = tokenizer(batch['student_prompt'], return_tensors='pt', padding=True, truncation=True, max_length=max_length)

        # Generate teacher outputs
        with torch.no_grad():
            teacher_outputs = teacher_model(**teacher_encoded).logits

        # Get student outputs
        student_outputs = student_model(**student_encoded).logits

        # Ensure the tensors are aligned in size
        min_length = min(student_outputs.shape[1], teacher_outputs.shape[1])
        student_outputs = student_outputs[:, :min_length, :]
        teacher_outputs = teacher_outputs[:, :min_length, :]

        # Calculate loss (Equation 1)
        loss = kl_divergence_loss(student_outputs, teacher_outputs)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
