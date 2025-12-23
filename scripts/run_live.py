import torch
import json
import os
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


MODEL_PATH = "/workspace/trained"
MAX_SEQ_LENGTH = 2048 # Forces RoPE Scaling for long interviews 4096

print(f" LOADING MODEL...")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

def analyze_candidate(role, jd, resume, transcript):
   
    input_text = f"### ROLE:\n{role}\n\n### JOB DESCRIPTION:\n{jd}\n\n### RESUME:\n{resume}\n\n### TRANSCRIPT:\n{transcript}"
    
    
    full_instruction = """You are a Senior Technical Recruiter. Evaluate the candidate based on a HOLISTIC assessment.

    CRITERIA:
    1. TECHNICAL (Primary): Does the candidate demonstrate the specific skills required in the JD? (Look for proof of usage, not just keywords).
    2. SOFT SKILLS (Secondary): Assess communication clarity, teamwork examples, and adaptability in the transcript.
    3. VERIFICATION: Reject vague claims. Select only if the candidate provides specific examples (STAR method).

    DECISION:
    - SELECT if they pass the Technical bar AND show good Soft Skills.
    - REJECT if they lack core Tech skills OR display major Red Flags (evasive, rude, lying).

    Output JSON: {'reasoning': 'Detailed analysis...', 'status': 'select/reject'}."""

    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": full_instruction},
        {"role": "user", "content": f"{full_instruction}\n\n{input_text}"}
    ], tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Low temp for logic consistency
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return response


if __name__ == "__main__":
    print("\n" + "="*60)
    print(" AI RECRUITER - JSON MODE")
    print("="*60)
    
    while True:
       
        print("\n Paste the path to your JSON file (e.g., sanjay.json)")
        json_path = input(" Path (or 'q' to quit): ").strip()
        
        if json_path.lower() == 'q':
            print(" Exiting.")
            break
            
        
        json_path = json_path.replace('"', '').replace("'", "")
        
        if not os.path.exists(json_path):
            print(" File not found! Check the name and try again.")
            continue
            
        try:
        
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            print(f"\n Analyzing {data.get('role', 'Candidate')}...")
            
            # Run the AI
            result = analyze_candidate(
                role=data.get('role', ''),
                jd=data.get('jd', ''),
                resume=data.get('resume', ''),
                transcript=data.get('transcript', '')
            )
            
           
            print("\n" + "-" * 50)
            print(" AI DECISION:")
            print("-" * 50)
            print(result)
            print("-" * 50)
            
        except json.JSONDecodeError:
            print(" Valid file found, but it contains INVALID JSON syntax.")
        except Exception as e:
            print(f" Error: {e}")