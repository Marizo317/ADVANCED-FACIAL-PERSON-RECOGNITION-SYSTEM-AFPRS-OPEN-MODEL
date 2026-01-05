# llm/app/generator.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import LLM_CFG

class Generator:
    def __init__(self):
        print(f"[LLM] Loading {LLM_CFG.MODEL}...")
        
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        ) if LLM_CFG.LOAD_4BIT else None
        
        self.tok = AutoTokenizer.from_pretrained(LLM_CFG.MODEL, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_CFG.MODEL,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        self.model.eval()
        print("[LLM] Model ready")
    
    def generate(self, age, gender, emotion, gaze, upper, lower, position, depth) -> str:
        ctx = []
        if position and depth:
            ctx.append(f"Position: {position}, {depth} distance")
        if upper:
            ctx.append(f"Upper clothing: {upper}")
        if lower:
            ctx.append(f"Lower clothing: {lower}")
        if age and gender:
            ctx.append(f"Appears ~{age} years old, {gender}")
        if emotion:
            ctx.append(f"Expression: {emotion}")
        if gaze:
            ctx.append(f"Looking: {gaze}")
        
        prompt = f"""<|im_start|>system
You are a professional analyst. Write brief 2-sentence descriptions.
<|im_end|>
<|im_start|>user
Describe this person:
{chr(10).join(ctx)}
<|im_end|>
<|im_start|>assistant
"""
        
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=LLM_CFG.MAX_TOKENS,
                temperature=LLM_CFG.TEMPERATURE,
                do_sample=True,
                pad_token_id=self.tok.eos_token_id
            )
        
        resp = self.tok.decode(out[0], skip_special_tokens=True)
        if "<|im_start|>assistant" in resp:
            resp = resp.split("<|im_start|>assistant")[-1]
        return resp.replace("<|im_end|>", "").strip()[:200]
