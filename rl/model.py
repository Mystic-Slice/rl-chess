from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

def get_model(model_id, model_type='lora'):
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if model_type == 'lora':
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, # type of task to train on
            r=16, # dimension of the smaller matrices
            lora_alpha=32, # scaling factor
            lora_dropout=0.1, # dropout of LoRA layers
            # target_modules=[
            #     "q_proj", "k_proj", "v_proj", 
            #     "o_proj", "gate_proj", "up_proj", "down_proj"
            # ],
            target_modules='all-linear',
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model, tokenizer
    elif model_type == 'full':
        return model, tokenizer
    else:
        raise ValueError(f"Invalid model_type. Got {model_type}. Expected ['lora', 'full']")

    return model, tokenizer