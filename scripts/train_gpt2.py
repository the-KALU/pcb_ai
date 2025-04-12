from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add padding token if it's missing
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset_path = "pcb_data.jsonl"
data = load_dataset("json", data_files=dataset_path, split="train")

# Tokenize examples
def tokenize(example):
    text = f"### Prompt:\n{example['prompt']}\n\n### Response:\n{example['completion']}"
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

tokenized_data = data.map(tokenize, batched=False)

# Set up training configuration
training_args = TrainingArguments(
    output_dir="./gpt2-pcb-model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    
    learning_rate=5e-5,
    fp16=False,
    report_to="none"  # disables wandb or other loggers
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 does not use masked language modeling
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    data_collator=data_collator
)

# Fine-tune the model!
trainer.train()

# SAVE MODEL CLEANLY
model.save_pretrained("./gpt2-pcb-model")
tokenizer.save_pretrained("./gpt2-pcb-model")