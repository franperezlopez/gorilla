# Based on code from https://colab.research.google.com/drive/1vk8i01apaSp59GVV2yInxOV15QwCwMrg
import argparse
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

DEFAULT_MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf"

def get_args() -> any:
    """
    Parses and returns the arguments specified by the user's command
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, help="The path at which the dataset is located")
    parser.add_argument("--output_model_name", type=str, help="The name of the output model")
    parser.add_argument("--base_model_name", type=str, default="DEFAULT_MODEL_NAME", help="The name of the base model to use for training")

    args = parser.parse_args()
    return args

def format_instruction(sample):
	# for i in range(len(sample['instruction'])):
	return f"""Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the task.

### Instruction:
{sample['prompt_instruction']}

### Input:
{sample['prompt_input']}

### Response:
{sample.get('prompt_cot_response', '')}
"""

def find_all_linear_names(model):
    import bitsandbytes as bnb
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def get_tokenizer_model(base_model_name: str = DEFAULT_MODEL_NAME, inference_mode: bool = False):
    # Tokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"  # Fix for fp16

    # Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,        
        device_map={"": 0},
    )

    base_model.config.use_cache = inference_mode

    return base_model, llama_tokenizer

def configure_model(base_model, llama_tokenizer, training_data):
    # LoRA Config
    peft_parameters = LoraConfig(
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=find_all_linear_names(base_model),
        r=32,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Training Params
    train_params = TrainingArguments(
        output_dir="./results_modified",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    # Trainer
    fine_tuning = SFTTrainer(
        model=base_model,
        train_dataset=training_data,
        max_seq_length=1024,
        peft_config=peft_parameters,
        formatting_func=format_instruction,
        packing=True,
        tokenizer=llama_tokenizer,
        args=train_params
    )

    return fine_tuning

def main(dataset_path: str, base_model_name: str, output_model_name: str):
    """
    Fine-tunes a model using the provided dataset and saves the trained model and tokenizer.

    Args:
        dataset_path (str): The path to the dataset.
        base_model_name (str): The name of the base model to use for fine-tuning.
        output_model_name (str): The name to use when saving the trained model and tokenizer.

    Returns:
        fine_tuning: The fine-tuning object containing the trained model and tokenizer.
    """
    training_data = load_from_disk(dataset_path)

    model, tokenizer = get_tokenizer_model(base_model_name)
    fine_tuning = configure_model(model, tokenizer, training_data)

    # Training
    fine_tuning.train()

    # Save Model and tokenizer
    fine_tuning.model.save_pretrained(output_model_name, safe_serialization=True)
    tokenizer.save_pretrained(output_model_name)
    return fine_tuning

if __name__ == "__main__":
    args = get_args()

    main(args.dataset_path, args.base_model_name, args.output_model_name)