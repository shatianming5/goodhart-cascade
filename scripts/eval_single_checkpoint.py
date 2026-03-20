#!/usr/bin/env python3
"""Evaluate a single checkpoint - for parallel execution."""
import os, sys, json, argparse
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from goodhart.eval.calibration import CalibrationEvaluator
from goodhart.eval.code_quality import CodeQualityEvaluator
import math


def load_model_and_tokenizer(base_model, adapter_path, device="cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device
    )
    if adapter_path and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()

    def generate_fn(prompt, return_logprobs=False):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=2048, temperature=0.0,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
                output_logits=return_logprobs, return_dict_in_generate=return_logprobs,
            )
        if return_logprobs:
            generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            logprobs_dict = {}
            if outputs.logits and len(outputs.logits) > 0:
                first_logits = outputs.logits[0][0]
                probs = torch.softmax(first_logits, dim=-1)
                for token_str in ["Yes", "yes", "No", "no"]:
                    token_id = tokenizer.encode(token_str, add_special_tokens=False)
                    if token_id:
                        p = probs[token_id[0]].item()
                        logprobs_dict[token_str] = math.log(p) if p > 0 else -100.0
            return response, logprobs_dict
        else:
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--step_name", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/trl_train.json")
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    step_dir = Path(args.results_dir) / args.step_name
    if (step_dir / "done.flag").exists():
        print("Already done: " + args.step_name)
        return

    # Load eval data
    with open(args.data_path) as f:
        data = json.load(f)
    eval_data = data[-args.n_eval:]
    problems = []
    for item in eval_data:
        gt = item["ground_truth"]
        if isinstance(gt, str):
            gt = json.loads(gt)
        prompt_text = item["prompt"]
        if isinstance(prompt_text, list):
            question = prompt_text[-1].get("content", "") if prompt_text else ""
        else:
            question = prompt_text
        problems.append({"question": question, "test_cases": gt, "starter_code": "", "difficulty": "unknown"})

    adapter = args.adapter_path if args.adapter_path else None
    print("Loading model for " + args.step_name + "...")
    generate_fn = load_model_and_tokenizer(args.base_model, adapter, args.device)

    step_dir.mkdir(parents=True, exist_ok=True)
    cal_eval = CalibrationEvaluator(problems, n_samples=4)
    qual_eval = CodeQualityEvaluator(problems)

    print("  Calibration...")
    try:
        cal_results = cal_eval.evaluate_logprob(generate_fn)
        with open(step_dir / "calibration.json", "w") as f:
            json.dump(cal_results, f, indent=2)
        print("  ECE=" + str(round(cal_results["ece"], 4)))
    except Exception as e:
        print("  Cal failed: " + str(e))

    print("  Code quality...")
    try:
        qual_results = qual_eval.evaluate(generate_fn)
        with open(step_dir / "code_quality.json", "w") as f:
            json.dump(qual_results, f, indent=2)
        print("  pylint=" + str(round(qual_results.get("pylint_score", 0), 2)))
    except Exception as e:
        print("  Qual failed: " + str(e))

    with open(step_dir / "done.flag", "w") as f:
        f.write("done")
    print("Done: " + args.step_name)


if __name__ == "__main__":
    main()
