from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
import torch

def get_model(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
        device_map=0
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, # type of task to train on
        r=16, # dimension of the smaller matrices
        lora_alpha=32, # scaling factor
        lora_dropout=0.1, # dropout of LoRA layers
        target_modules=[
            "q_proj", "k_proj", "v_proj", 
            "o_proj", "gate_proj", "up_proj", "down_proj"
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer