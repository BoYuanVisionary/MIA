from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig
from numpy import percentile


# Hugging Face model id
model_id = "meta-llama/Llama-3.2-1B-Instruct" # replace with your model id

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='cuda:0',
    use_cache=False,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left' # to prevent errors with FA
tokenizer.truncation_side = 'left' # to prevent cutting off last generation

# Load jsonl data from disk
train_dataset = load_dataset("json", data_files="train_dataset.json", split="train")
eval_dataset = load_dataset("json", data_files="test_dataset.json", split="train")

# lets find the p95 length of the prompt
prompt_length = int(percentile([len(tokenizer(x)["input_ids"]) for x in train_dataset["prompt"]], 95))
max_seq_length_chosen = int(percentile([len(tokenizer(x["prompt"] + x["chosen"])["input_ids"]) for x in train_dataset], 95))
max_seq_length_rejected = int(percentile([len(tokenizer(x["prompt"] + x["rejected"])["input_ids"]) for x in train_dataset], 95))
max_seq_length = max(max_seq_length_chosen, max_seq_length_rejected)

# filter datasets to remove samples that are too long
train_dataset = train_dataset.filter(lambda x: len(tokenizer(x["prompt"] + x["chosen"])["input_ids"]) <= max_seq_length)
eval_dataset = eval_dataset.filter(lambda x: len(tokenizer(x["prompt"] + x["chosen"])["input_ids"]) <= max_seq_length)
print(f"len(train_dataset): {len(train_dataset)}") # exact number of samples in training set
print(f"len(eval_dataset): {len(eval_dataset)}") # exact number of samples in testing set

# Up the lengths to next multiple of 2, why 2? Don't know
prompt_length = ((prompt_length + 1) // 2) * 2
max_seq_length = ((max_seq_length + 1) // 2) * 2
print(f"p95 prompt length: {prompt_length}")
print(f"p95 prompt + chosen length: {max_seq_length}")

dpo_args = {
    "beta": 0.1,                            # The beta factor in DPO loss. Higher beta means less divergence
    "loss_type": "sigmoid"                  # The loss type for DPO.
}

args = DPOConfig(
    output_dir="llama3.2-1B-dpo-v1",        # directory to save and repository id
    num_train_epochs=1,                     # number of training epochs
    per_device_train_batch_size=6,          # batch size per device during training
    per_device_eval_batch_size=4,           # batch size for evaluation
    gradient_accumulation_steps=1,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    learning_rate=5e-5,                     # 10x higher LR than QLoRA paper
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.1,                       # warmup ratio based on QLoRA paper
    lr_scheduler_type="cosine",             # use cosine learning rate scheduler
    logging_steps=25,                       # log every 25 steps
    save_steps=500,                         # when to save checkpoint
    save_total_limit=2,                     # limit the total amount of checkpoints
    evaluation_strategy="steps",            # evaluate based on steps
    eval_steps=700,                         # when to evaluate every 700 steps
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    push_to_hub=True,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
    remove_unused_columns=False,
    max_length=max_seq_length,
    max_prompt_length=prompt_length,
    loss_type=dpo_args["loss_type"],
)

peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)

trainer = DPOTrainer(
    model,
    ref_model=None, # set to none since we use peft
    peft_config=peft_config,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    beta=dpo_args["beta"],

)

# start training, the model will be automatically saved to the hub and the output directory
trainer.train()
# save model at the end of training
trainer.save_model()