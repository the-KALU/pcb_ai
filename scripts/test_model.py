from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel

# Step 1: Load tokenizer from base GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Step 2: Load fine-tuned model from checkpoint
model = GPT2LMHeadModel.from_pretrained("./gpt2-pcb-model/checkpoint-36")  # 👈 adjust if checkpoint folder is different

# Step 3: Set up the generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Step 4: Test prompt
prompt = "Design a 2-layer PCB with an ESP32, a USB-C power input, an LDO regulator, and two I2C temperature sensors."

# Step 5: Generate and print response
output = generator(f"### Prompt:\n{prompt}\n\n### Response:\n", 
                   max_length=200, 
                   do_sample=True, 
                   temperature=0.9)

print("\n🔧 AI-Generated PCB Design Steps:\n")
print(output[0]["generated_text"])
