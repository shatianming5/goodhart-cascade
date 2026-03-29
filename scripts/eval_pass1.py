#!/usr/bin/env python3
"""Evaluate pass@1 on validation set."""
import os, sys, json, argparse, torch
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from goodhart.rewards.test_passing import extract_code_from_response
from goodhart.utils.code_exec import run_all_tests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--data_path", type=str, default="data/trl_val_filtered.json")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=args.device
    )
    if args.adapter_path and os.path.exists(os.path.join(args.adapter_path, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
    model.eval()

    with open(args.data_path) as f:
        data = json.load(f)

    passed = 0
    total = len(data)
    for i, item in enumerate(data):
        prompt = item["prompt"]
        if isinstance(prompt, list):
            prompt_text = prompt[-1].get("content", "") if prompt else ""
        else:
            prompt_text = prompt
        messages = [{"role": "user", "content": prompt_text}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=2048, temperature=1.0, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        code = extract_code_from_response(resp)
        gt = item["ground_truth"]
        if isinstance(gt, str):
            gt = json.loads(gt)
        try:
            ok, _, _ = run_all_tests(code, gt, timeout=5, max_memory_mb=512)
            if ok:
                passed += 1
        except:
            pass
        if (i+1) % 10 == 0:
            print(f"  [{args.name}] {i+1}/{total}: pass={passed}/{i+1} ({passed/(i+1)*100:.1f}%)")

    result = {"name": args.name, "pass_at_1": passed/total, "passed": passed, "total": total}
    os.makedirs("results/pass1", exist_ok=True)
    with open(f"results/pass1/{args.name}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.name}] FINAL: pass@1 = {passed}/{total} = {passed/total:.4f}")

if __name__ == "__main__":
    main()
