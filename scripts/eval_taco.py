#!/usr/bin/env python3
"""Evaluate base + R1-R5 v2 on TACO sweet spot (held-out 200 problems). Sequential on GPU 0."""
import json, os, shutil, subprocess, sys, time
from huggingface_hub import hf_hub_download, list_repo_files

HF_TOKEN = "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"
EVAL_DIR = "/root/goodhart-cascade/results/eval_taco"
GPU = 0
SKIP = {"optimizer.pt","rng_state.pth","scheduler.pt","trainer_state.json","training_args.bin"}

JOBS = [
    ("base", "Qwen/Qwen2.5-Coder-7B", None),
    ("7B_R1_v2", "Tommysha/goodhart-cascade-7B_R1_v2", "checkpoint-1000"),
    ("7B_R2_v2", "Tommysha/goodhart-cascade-7B_R2_v2", "checkpoint-1000"),
    ("7B_R3_v2", "Tommysha/goodhart-cascade-7B_R3_v2", "checkpoint-1000"),
    ("7B_R4_v2", "Tommysha/goodhart-cascade-7B_R4_v2", "checkpoint-1000"),
    ("7B_R5_v2", "Tommysha/goodhart-cascade-7B_R5_v2", "checkpoint-1000"),
    # Peak checkpoints
    ("7B_R1_v2_p500", "Tommysha/goodhart-cascade-7B_R1_v2", "checkpoint-500"),
    ("7B_R2_v2_p600", "Tommysha/goodhart-cascade-7B_R2_v2", "checkpoint-600"),
    ("7B_R3_v2_p300", "Tommysha/goodhart-cascade-7B_R3_v2", "checkpoint-300"),
    ("7B_R4_v2_p300", "Tommysha/goodhart-cascade-7B_R4_v2", "checkpoint-300"),
    ("7B_R5_v2_p500", "Tommysha/goodhart-cascade-7B_R5_v2", "checkpoint-500"),
]

os.chdir("/root/goodhart-cascade")
os.makedirs(EVAL_DIR, exist_ok=True)

for idx, (label, repo, sub) in enumerate(JOBS):
    rp = os.path.join(EVAL_DIR, f"{label}.json")
    if os.path.exists(rp):
        r = json.load(open(rp))
        print(f"[{idx+1}/{len(JOBS)}] {label}: done (Pass@1={r['pass_at_1']:.1%})", flush=True)
        continue

    local = f"results/tmp_taco_{label}"
    try:
        print(f"\n[{idx+1}/{len(JOBS)}] {label}", flush=True)
        if sub:
            files = list_repo_files(repo, token=HF_TOKEN)
            for f in files:
                if f.startswith(sub+"/") and f.split("/")[-1] not in SKIP:
                    hf_hub_download(repo, f, token=HF_TOKEN, local_dir=local)
            mp = os.path.join(local, sub)
        else:
            from huggingface_hub import snapshot_download
            snapshot_download(repo, token=HF_TOKEN, local_dir=local)
            mp = local
        print(f"  Downloaded", flush=True)

        sc = f'''
import json,os,sys,gc,torch,signal
os.environ["CUDA_VISIBLE_DEVICES"]="{GPU}"
sys.path.insert(0,"/root/goodhart-cascade")
from vllm import LLM, SamplingParams
from src.evaluation.evaluator import extract_code
from src.training.trl_grpo_trainer import _batch_test
from src.rewards.all_metrics import compute_all_metrics

# Load eval data (last 200 problems as held-out)
with open("data/sweet_spot_7B.json") as f:
    raw = json.load(f)
eval_data = raw[-200:]

# Build prompts
prompts = []
for item in eval_data:
    p = ("Write a Python solution for the following problem. "
         "Only output the code, no explanations.\\n\\n"
         f"Problem:\\n{{item['prompt']}}\\n")
    if item.get("starter_code"):
        p += f"\\nStarter code:\\n{{item['starter_code']}}\\n"
    p += "\\nSolution:\\n```python\\n"
    prompts.append(p)

test_cases = []
for item in eval_data:
    tc = item.get("test_cases", {{}})
    if isinstance(tc, str): tc = json.loads(tc)
    test_cases.append(tc)

# Load model
llm = LLM(model="{mp}", trust_remote_code=True, gpu_memory_utilization=0.85, max_model_len=4096)
greedy = SamplingParams(n=1, temperature=0.0, max_tokens=1024, stop=["```\\n","\\n\\n\\n"])
sample = SamplingParams(n=8, temperature=0.8, top_p=0.95, max_tokens=1024, stop=["```\\n","\\n\\n\\n"])

# Pass@1 (greedy)
g_out = llm.generate(prompts, greedy)
g_codes = [extract_code(o.outputs[0].text) for o in g_out]
g_results = _batch_test(g_codes, test_cases)
pass1 = sum(g_results)/len(g_results)

# Pass@8 (sampling)
s_out = llm.generate(prompts, sample)
pass8 = 0
for out, tc in zip(s_out, test_cases):
    codes = [extract_code(o.text) for o in out.outputs]
    res = _batch_test(codes, [tc]*len(codes))
    if any(res): pass8 += 1
pass8 /= len(eval_data)

# Quality metrics on greedy outputs
metrics = [compute_all_metrics(c) for c in g_codes if c.strip()]
def sm(k):
    v = [m[k] for m in metrics if k in m]
    return sum(v)/len(v) if v else 0

r = {{
    "label": "{label}",
    "pass_at_1": pass1,
    "pass_at_k": pass8,
    "n_eval": len(eval_data),
    "pylint": sm("pylint")*10,
    "complexity": sm("complexity_raw"),
    "comment_pct": sm("comment_ratio_raw")*100,
    "duplication_pct": sm("duplication_ratio_raw")*100,
    "type_hint_pct": sm("type_hint")*100,
    "avg_func_length": sm("avg_func_length"),
    "magic_numbers": sm("magic_numbers"),
    "nesting_depth": sm("nesting_depth"),
    "dead_code_pct": sm("dead_code")*100,
    "naming_length": sm("naming_length"),
}}
json.dump(r, open("{rp}","w"), indent=2)
del llm; gc.collect(); torch.cuda.empty_cache()
'''
        ef = f"/tmp/taco_{label}.py"
        open(ef,"w").write(sc)
        t0 = time.time()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(GPU)}
        proc = subprocess.run([sys.executable, ef], capture_output=True, text=True, timeout=1800, env=env)
        el = time.time()-t0
        os.remove(ef)
        if proc.returncode != 0:
            print(f"  ERROR: {proc.stderr[-500:]}", flush=True)
        elif os.path.exists(rp):
            r = json.load(open(rp))
            print(f"  Pass@1={r['pass_at_1']:.1%} Pass@8={r['pass_at_k']:.1%} Pylint={r['pylint']:.1f} Cmplx={r['complexity']:.1f} ({el:.0f}s)", flush=True)
    finally:
        if os.path.exists(local): shutil.rmtree(local)

# Summary
print("\n" + "="*110)
print(f"{'Label':<18} {'Pass@1':>7} {'Pass@8':>7} {'Pylint':>7} {'Cmplx':>7} {'Cmt%':>7} {'Dup%':>6} {'FuncL':>6} {'Magic':>6} {'Nest':>5} {'Name':>5}")
print("-"*110)
order = [j[0] for j in JOBS]
for lb in order:
    rp = os.path.join(EVAL_DIR, f"{lb}.json")
    if not os.path.exists(rp): continue
    r = json.load(open(rp))
    print(f"{lb:<18} {r['pass_at_1']:>6.1%} {r['pass_at_k']:>6.1%} {r['pylint']:>7.1f} {r['complexity']:>7.1f} "
          f"{r.get('comment_pct',0):>6.1f}% {r.get('duplication_pct',0):>5.1f}% "
          f"{r.get('avg_func_length',0):>6.1f} {r.get('magic_numbers',0):>6.1f} {r.get('nesting_depth',0):>5.1f} {r.get('naming_length',0):>5.1f}")
print("="*110)

# Save combined
all_r = {}
for lb in order:
    rp = os.path.join(EVAL_DIR, f"{lb}.json")
    if os.path.exists(rp): all_r[lb] = json.load(open(rp))
json.dump(all_r, open(os.path.join(EVAL_DIR, "all_taco_results.json"), "w"), indent=2)
print(f"\nSaved to {EVAL_DIR}/all_taco_results.json")
