import os
import sys
import json
import torch
import random
import numpy as np
import shutil
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback, DataCollatorForSeq2Seq

# 1. ENVIRONMENT SETUP
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true" 

# 2. CONFIGURATION
CONFIG = {
    "MODEL_PATH": "/workspace/llama-3.2-3b-4bit",
    
    # CLEAN DATA PATHS
    "TRAIN_DATA": "/workspace/data/train_70.json",
    "VAL_DATA":   "/workspace/data/val_15.json",
    "TEST_DATA":  "/workspace/data/test_15.json",
    "OUTPUT_DIR": "/workspace/trained",
    
    "MAX_SEQ_LENGTH": 2048,   
    "LEARNING_RATE": 5e-5,    
    "BATCH_SIZE": 2,          
    "GRADIENT_ACCUMULATION": 8, 
    "EPOCHS": 4,              
    "WARMUP_STEPS": 100,      
    "LORA_R": 16,             
    "LORA_ALPHA": 16,         
    "LORA_DROPOUT": 0.05,     
    "SEED": 3407,
}

Path(CONFIG["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)


def find_latest_checkpoint(output_dir):
    if "--resume" not in sys.argv:
        print("Starting Fresh (Use --resume to continue)")
        return None
    
    checkpoints = sorted(Path(output_dir).glob("checkpoint-*"), key=lambda x: int(x.name.split('-')[1]))
    if checkpoints:
        print(f"Resuming from: {checkpoints[-1].name}")
        return str(checkpoints[-1])
    return None

def formatting_prompts_func(tokenizer, examples):
    texts = []
    system_prompt = """You are a Senior Technical Recruiter. Evaluate the candidate based on a HOLISTIC assessment.

    CRITERIA:
    1. TECHNICAL (Primary): Does the candidate demonstrate the specific skills required in the JD? (Look for proof of usage, not just keywords).
    2. SOFT SKILLS (Secondary): Assess communication clarity, teamwork examples, and adaptability in the transcript.
    3. VERIFICATION: Reject vague claims. Select only if the candidate provides specific examples (STAR method).

    DECISION:
    - SELECT if they pass the Technical bar AND show good Soft Skills.
    - REJECT if they lack core Tech skills OR display major Red Flags (evasive, rude, lying).

    Output JSON: {'reasoning': 'Detailed analysis...', 'status': 'select/reject'}."""
    for instruction, input_text, output_text in zip(examples["instruction"], examples["input"], examples["output"]):
        try:
            data = json.loads(output_text)
            cot_data = {
                "reasoning": data.get("reasoning", data.get("reason", "No reasoning provided.")),
                "status": data.get("status", "error")
            }
            new_output_text = json.dumps(cot_data)
        except:
            new_output_text = output_text

        convo = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{instruction}\n\n{input_text}"},
            {"role": "assistant", "content": new_output_text}
        ]
        text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        texts.append(text)
    return {"text": texts}

def evaluate_final(model, tokenizer, test_path):
    print("\n" + "="*60)
    print(" FINAL VALIDATION (Matched Prompt + Smart Parsing)")
    print("="*60)
    
    FastLanguageModel.for_inference(model)
    with open(test_path, 'r') as f:
        data = json.load(f)
    
    y_true = []
    y_pred = []
    
    print(f"Testing {len(data)} samples...")

    system_prompt = """You are a Senior Technical Recruiter. Evaluate the candidate based on a HOLISTIC assessment.

    CRITERIA:
    1. TECHNICAL (Primary): Does the candidate demonstrate the specific skills required in the JD? (Look for proof of usage, not just keywords).
    2. SOFT SKILLS (Secondary): Assess communication clarity, teamwork examples, and adaptability in the transcript.
    3. VERIFICATION: Reject vague claims. Select only if the candidate provides specific examples (STAR method).

    DECISION:
    - SELECT if they pass the Technical bar AND show good Soft Skills.
    - REJECT if they lack core Tech skills OR display major Red Flags (evasive, rude, lying).

    Output JSON: {'reasoning': 'Detailed analysis...', 'status': 'select/reject'}."""
    for item in tqdm(data):
        try:
            gt_json = json.loads(item['output'])
            gt_status = gt_json.get('status', '').lower().strip()
        except:
            continue

        prompt = tokenizer.apply_chat_template([
            {
                "role": system_prompt, 
                "content": "You are an expert AI Recruiter. Analyze the Role, JD, Resume, and Transcript. Output JSON with format: {'reasoning': '...', 'status': 'select/reject'}."
            },
            {"role": "user", "content": f"{item['instruction']}\n\n{item['input']}"}
        ], tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        pred_status = "error"
        lower_resp = response.lower()
        if '"status": "select"' in lower_resp or "'status': 'select'" in lower_resp:
            pred_status = "select"
        elif '"status": "reject"' in lower_resp or "'status': 'reject'" in lower_resp:
            pred_status = "reject"
        
        if pred_status == "error":
             s_idx = lower_resp.rfind("select")
             r_idx = lower_resp.rfind("reject")
             if s_idx > r_idx: pred_status = "select"
             if r_idx > s_idx: pred_status = "reject"

        y_true.append(gt_status)
        y_pred.append(pred_status)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=["select", "reject"], digits=4)
    
    print("\n FINAL RESULTS:")
    print(f"Accuracy:  {acc:.2%}")
    print(report)

def main():
    print("="*60)
    print(f" TRAINING START: Target > 90% Accuracy")
    

  
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = CONFIG["MODEL_PATH"],
        max_seq_length = CONFIG["MAX_SEQ_LENGTH"],
        dtype = None,
        load_in_4bit = True,
    )

   
    model = FastLanguageModel.get_peft_model(
        model,
        r = CONFIG["LORA_R"],
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = CONFIG["LORA_ALPHA"],
        lora_dropout = CONFIG["LORA_DROPOUT"],
        bias = "none",
        use_gradient_checkpointing = "unsloth", 
        random_state = CONFIG["SEED"],
        use_rslora = True, 
    )

   
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    
    # 4. Load Data
    print(f"\n Loading Clean Data...")
    def load_json_to_dataset(path):
        with open(path, 'r') as f:
            data = json.load(f)
        return Dataset.from_list(data)

    train_dataset = load_json_to_dataset(CONFIG["TRAIN_DATA"])
    eval_dataset  = load_json_to_dataset(CONFIG["VAL_DATA"])

    print(" Formatting Prompts...")
    train_dataset = train_dataset.map(lambda x: formatting_prompts_func(tokenizer, x), batched=True, num_proc=2)
    eval_dataset = eval_dataset.map(lambda x: formatting_prompts_func(tokenizer, x), batched=True, num_proc=2)

 
    training_args = SFTConfig(
        output_dir = CONFIG["OUTPUT_DIR"],
        per_device_train_batch_size = CONFIG["BATCH_SIZE"],
        gradient_accumulation_steps = CONFIG["GRADIENT_ACCUMULATION"],
        warmup_steps = CONFIG["WARMUP_STEPS"],
        num_train_epochs = CONFIG["EPOCHS"],
        learning_rate = CONFIG["LEARNING_RATE"],
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = CONFIG["SEED"],
        dataset_num_proc = 2,
        dataloader_num_workers = 0,
        eval_strategy = "epoch",   
        save_strategy = "epoch",  
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        dataset_text_field = "text",
        max_seq_length = CONFIG["MAX_SEQ_LENGTH"],
        packing = False, 
        report_to = "none",
    )

   
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset, 
        dataset_text_field = "text",
        max_seq_length = CONFIG["MAX_SEQ_LENGTH"],
        dataset_num_proc = 2,
        args = training_args,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

 
    resume_path = find_latest_checkpoint(CONFIG["OUTPUT_DIR"])
    print(f"\n STARTING TRAINING...")
    trainer.train(resume_from_checkpoint=resume_path)

   
    print(f"\n SAVING BEST MODEL...")
    model.save_pretrained(CONFIG["OUTPUT_DIR"])
    tokenizer.save_pretrained(CONFIG["OUTPUT_DIR"])
    
    
    evaluate_final(model, tokenizer, CONFIG["TEST_DATA"]) 

   
    print(f"\n Converting to GGUF...")
    try:
        model.save_pretrained_gguf(CONFIG["OUTPUT_DIR"] + "_q4km", tokenizer, quantization_method="q4_k_m")
    except Exception as e:
        print(f" GGUF Conversion Warning: {e}")

if __name__ == "__main__":
    main()