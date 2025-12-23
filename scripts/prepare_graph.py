import pandas as pd
import json
import re
from collections import Counter
from sklearn.model_selection import train_test_split


INPUT_CSV = "/workspace/data/dataset.csv"

# OUTPUTS (The 3 Splits)
OUTPUT_TRAIN = "/workspace/data/train_70.json"
OUTPUT_VAL   = "/workspace/data/val_15.json"
OUTPUT_TEST  = "/workspace/data/test_15.json"

MAX_LEN = 3000 # Truncate long text to save memory

def scrub_pii(text):
    text = str(text)
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    text = re.sub(r'\d{3}[-.\s]??\d{3}[-.\s]??\d{4}', '[PHONE]', text)
    return text

def main():
    print(f" Reading CSV: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
    except:
        df = pd.read_csv(r"D:\PythonProject3\data\dataset.csv") # Local fallback

    print(f"   Total Raw Records: {len(df)}")

    # 1. CLEANING & FILTERING
    processed_data = []
    skipped_count = 0
    used_reasons = Counter()

    print("\n⚙  Processing & De-Duplicating...")
    
    for idx, row in df.iterrows():
        # Clean reasoning text
        reason = str(row['Reason_for_decision']).strip()
        
        #  ANTI-SPAM FILTER

        if used_reasons[reason] >= 10:
            skipped_count += 1
            continue
        used_reasons[reason] += 1
        
        # Format Input (Direct Text)
        role = str(row['Role'])
        jd = scrub_pii(row['Job_Description'])[:1000]
        resume = scrub_pii(row['Resume'])[:1500]
        transcript = scrub_pii(row['Transcript'])[:MAX_LEN]
        
        input_text = f"""### ROLE:
{role}

### JOB DESCRIPTION:
{jd}...

### RESUME:
{resume}...

### TRANSCRIPT:
{transcript}"""

        # Format Output (Reasoning First -> CoT)
        output_json = {
            "reasoning": reason,
            "status": str(row['decision']).lower().strip()
        }
        system_prompt = """You are a Senior Technical Recruiter. Evaluate the candidate based on a HOLISTIC assessment.

        CRITERIA:
        1. TECHNICAL (Primary): Does the candidate demonstrate the specific skills required in the JD? (Look for proof of usage, not just keywords).
        2. SOFT SKILLS (Secondary): Assess communication clarity, teamwork examples, and adaptability in the transcript.
        3. VERIFICATION: Reject vague claims. Select only if the candidate provides specific examples (STAR method).

        DECISION:
        - SELECT if they pass the Technical bar AND show good Soft Skills.
        - REJECT if they lack core Tech skills OR display major Red Flags (evasive, rude, lying).

        Output JSON: {'reasoning': 'Detailed analysis...', 'status': 'select/reject'}."""
        processed_data.append({
            "instruction": system_prompt,
            "input": input_text,
            "output": json.dumps(output_json)
        })

    print(f"✂  Removed {skipped_count} spam samples.")
    print(f" Total Clean Samples: {len(processed_data)}")

    # 2. THE 70 / 15 / 15 SPLIT
    # First, split off the 15% Final Test Set
    train_val, test = train_test_split(processed_data, test_size=0.15, random_state=42)
    
    # Next, split the remaining 85% into Train (70% total) and Val (15% total)
    # 0.176 of 85% is approx 15% of the total
    train, val = train_test_split(train_val, test_size=0.1765, random_state=42)
    
    # 3. SAVE
    with open(OUTPUT_TRAIN, 'w') as f: json.dump(train, f, indent=2)
    with open(OUTPUT_VAL, 'w') as f: json.dump(val, f, indent=2)
    with open(OUTPUT_TEST, 'w') as f: json.dump(test, f, indent=2)
    
    print("\n DATASET SPLIT COMPLETED:")
    print(f"   1. TRAIN (70%): {OUTPUT_TRAIN} ({len(train)} samples) -> Use for Training")
    print(f"   2. VAL   (15%): {OUTPUT_VAL}   ({len(val)} samples)   -> Use for Eval Steps")
    print(f"   3. TEST  (15%): {OUTPUT_TEST}  ({len(test)} samples)  -> Use for Final Accuracy")

if __name__ == "__main__":
    main()