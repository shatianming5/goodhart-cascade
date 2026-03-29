#!/usr/bin/env python3
"""Evaluate 1.5B + 14B on TACO + HumanEval. Sequential per scale on assigned GPU."""
import json, os, sys, gc, torch, signal, shutil, time
sys.path.insert(0, "/root/RD-agent")
from datasets import load_dataset
from vllm import LLM, SamplingParams
from src.evaluation.evaluator import extract_code, run_mbpp_tests
from src.training.trl_grpo_trainer import _batch_test
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

token = "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"
skip = {"optimizer.pt","rng_state.pth","scheduler.pt","trainer_state.json","training_args.bin"}

GPU = int(sys.argv[1])  # GPU ID
SCALE = sys.argv[2]     # "1.5B" or "14B"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
os.chdir("/root/RD-agent")

OUT_DIR = f"results/eval_{SCALE}"
os.makedirs(OUT_DIR, exist_ok=True)

# Load eval data
if SCALE == "1.5B":
    with open("data/sweet_spot_1.5B.json") as f: taco_raw = json.load(f)
    base_model = "Qwen/Qwen2.5-Coder-1.5B"
    MODELS = [
        ("base_1.5B", base_model, None),
        ("1.5B_R1", "Tommysha/goodhart-cascade-1.5B_R1_v2", "checkpoint-1000"),
        ("1.5B_R2", "Tommysha/goodhart-cascade-1.5B_R2_v2", "checkpoint-1000"),
        ("1.5B_R3", "Tommysha/goodhart-cascade-1.5B_R3_v2", "checkpoint-1000"),
        ("1.5B_R4", "Tommysha/goodhart-cascade-1.5B_R4_v2", "checkpoint-1000"),
        ("1.5B_R5", "Tommysha/goodhart-cascade-1.5B_R5_v2", "checkpoint-1000"),
    ]
else:
    with open("data/sweet_spot_14B.json") as f: taco_raw = json.load(f)
    base_model = "Qwen/Qwen2.5-Coder-14B"
    MODELS = [
        ("base_14B", base_model, None),
        ("14B_R1", "Tommysha/goodhart-cascade-14B_R1_v2", "checkpoint-1000"),
        ("14B_R2", "Tommysha/goodhart-cascade-14B_R2_v2", "checkpoint-1000"),
        ("14B_R3", "Tommysha/goodhart-cascade-14B_R3_v2", "checkpoint-1000"),
        ("14B_R4", "Tommysha/goodhart-cascade-14B_R4_v2", "checkpoint-1000"),
    ]

taco_eval = taco_raw[-200:]
taco_prompts = []
taco_tc = []
for item in taco_eval:
    p = ("Write a Python solution for the following problem. "
         "Only output the code, no explanations.\n\n"
         f"Problem:\n{item['prompt']}\n")
    if item.get("starter_code"): p += f"\nStarter code:\n{item['starter_code']}\n"
    p += "\nSolution:\n```python\n"
    taco_prompts.append(p)
    tc = item.get("test_cases", {})
    if isinstance(tc, str): tc = json.loads(tc)
    taco_tc.append(tc)

he_ds = load_dataset("openai/openai_humaneval", split="test")
he_prompts = [item["prompt"] for item in he_ds]
he_tests = [item["test"] for item in he_ds]
he_entry = [item["entry_point"] for item in he_ds]
greedy = SamplingParams(n=1, temperature=0.0, max_tokens=1024, stop=["```\n","\n\n\n"])
he_greedy = SamplingParams(n=1, temperature=0.0, max_tokens=1024, stop=["\nclass ","\ndef ","\n#","\nif __name__"])

def he_test(prompt, completion, test_code, entry_point, timeout=5):
    def _h(s,f): raise TimeoutError()
    old = signal.signal(signal.SIGALRM, _h)
    try:
        signal.alarm(timeout)
        exec(prompt + completion + "\n" + test_code + f"\ncheck({entry_point})", {})
        signal.alarm(0); return True
    except:
        signal.alarm(0); return False
    finally:
        signal.signal(signal.SIGALRM, old)

results = {}
for label, repo, sub in MODELS:
    rp = os.path.join(OUT_DIR, f"{label}.json")
    if os.path.exists(rp):
        results[label] = json.load(open(rp))
        print(f"[{label}] skip (done: TACO={results[label]['taco_pass1']:.1%} HE={results[label]['he_pass1']:.1%})", flush=True)
        continue

    local = f"results/tmp_{label}"
    try:
        print(f"[{label}] downloading...", flush=True)
        if sub:
            files = list_repo_files(repo, token=token)
            for f in files:
                if f.startswith(sub+"/") and f.split("/")[-1] not in skip:
                    hf_hub_download(repo, f, token=token, local_dir=local)
            mp = os.path.join(local, sub)
        else:
            snapshot_download(repo, token=token, local_dir=local)
            mp = local

        llm = LLM(model=mp, trust_remote_code=True, gpu_memory_utilization=0.85, max_model_len=4096)

        # TACO
        t_out = llm.generate(taco_prompts, greedy)
        t_codes = [extract_code(o.outputs[0].text) for o in t_out]
        t_res = _batch_test(t_codes, taco_tc)
        taco_p1 = sum(t_res)/len(t_res)

        # HumanEval
        h_out = llm.generate(he_prompts, he_greedy)
        he_p1 = sum(he_test(p,o.outputs[0].text,t,e) for p,o,t,e in zip(he_prompts,h_out,he_tests,he_entry)) / len(he_ds)

        r = {"taco_pass1": taco_p1, "he_pass1": he_p1, "label": label}
        results[label] = r
        json.dump(r, open(rp,"w"), indent=2)
        print(f"[{label}] TACO={taco_p1:.1%}  HE={he_p1:.1%}", flush=True)
        del llm; gc.collect(); torch.cuda.empty_cache()
    finally:
        if os.path.exists(local): shutil.rmtree(local)

# Summary
print(f"\n{'='*50}")
print(f"{SCALE} Results")
print(f"{'Model':<12} {'TACO P@1':>10} {'HE P@1':>10}")
print(f"{'-'*50}")
for label in [m[0] for m in MODELS]:
    r = results.get(label, {})
    print(f"{label:<12} {r.get('taco_pass1',0):>9.1%} {r.get('he_pass1',0):>9.1%}")

# Theory check
base_key = [m[0] for m in MODELS][0]
r1_key = [m[0] for m in MODELS][1]
if r1_key in results and "R4" in str(MODELS):
    r4_key = [m[0] for m in MODELS if "R4" in m[0]][0]
    r1_he = results[r1_key]["he_pass1"]
    r4_he = results[r4_key]["he_pass1"]
    print(f"\nTheory check: R4 should have highest tax (comment Cov<0)")
    print(f"  R1 HE: {r1_he:.1%}, R4 HE: {r4_he:.1%}, tax: {r1_he-r4_he:+.1%}")

json.dump(results, open(os.path.join(OUT_DIR, f"all_{SCALE}_results.json"),"w"), indent=2)
