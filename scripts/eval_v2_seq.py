#!/usr/bin/env python3
"""Sequential eval on GPU 2. Smart download: skip optimizer.pt."""
import json, os, shutil, subprocess, sys, time
from huggingface_hub import hf_hub_download, list_repo_files

HF_TOKEN = "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"
EVAL_DIR = "/root/goodhart-cascade/results/eval_v2"
GPU = 2
SKIP = {"optimizer.pt", "rng_state.pth", "scheduler.pt", "trainer_state.json", "training_args.bin"}

JOBS = [
    ("7B_R1_v2_final", "Tommysha/goodhart-cascade-7B_R1_v2", "checkpoint-1000"),
    ("7B_R2_v2_final", "Tommysha/goodhart-cascade-7B_R2_v2", "checkpoint-1000"),
    ("7B_R3_v2_final", "Tommysha/goodhart-cascade-7B_R3_v2", "checkpoint-1000"),
    ("7B_R4_v2_final", "Tommysha/goodhart-cascade-7B_R4_v2", "checkpoint-1000"),
    ("7B_R1_v2_peak",  "Tommysha/goodhart-cascade-7B_R1_v2", "checkpoint-500"),
    ("7B_R2_v2_peak",  "Tommysha/goodhart-cascade-7B_R2_v2", "checkpoint-600"),
    ("7B_R3_v2_peak",  "Tommysha/goodhart-cascade-7B_R3_v2", "checkpoint-300"),
    ("7B_R4_v2_peak",  "Tommysha/goodhart-cascade-7B_R4_v2", "checkpoint-300"),
]

os.chdir("/root/goodhart-cascade")
os.makedirs(EVAL_DIR, exist_ok=True)

for idx, (label, repo, sub) in enumerate(JOBS):
    rp = os.path.join(EVAL_DIR, f"{label}_results.json")
    if os.path.exists(rp):
        r = json.load(open(rp))
        print(f"[{idx+1}/{len(JOBS)}] {label}: done (Pass@1={r['pass_at_1']:.1%})", flush=True)
        continue

    local = f"results/tmp_ev_{label}"
    try:
        print(f"\n[{idx+1}/{len(JOBS)}] {label}", flush=True)
        files = list_repo_files(repo, token=HF_TOKEN)
        ckpt_files = [f for f in files if f.startswith(sub+"/") and f.split("/")[-1] not in SKIP]
        print(f"  Downloading {len(ckpt_files)} files from {sub}...", flush=True)
        for f in ckpt_files:
            hf_hub_download(repo, f, token=HF_TOKEN, local_dir=local)
        mp = os.path.join(local, sub)
        sz = sum(os.path.getsize(os.path.join(dp,fn)) for dp,_,fns in os.walk(mp) for fn in fns)/(1024**3)
        print(f"  Downloaded {sz:.1f} GB", flush=True)

        sc = f'''
import json,os,sys,gc,torch
os.environ["CUDA_VISIBLE_DEVICES"]="{GPU}"
sys.path.insert(0,"/root/goodhart-cascade")
from src.evaluation.evaluator import Evaluator,load_classeval,build_classeval_prompt,extract_code
from src.rewards.all_metrics import compute_all_metrics
from vllm import SamplingParams
ev=Evaluator(model_path="{mp}",n_mbpp=100,n_classeval=50,n_samples=8,gpu_memory_utilization=0.85)
r=ev.evaluate_all()
ce=load_classeval(50);cp=[build_classeval_prompt(p) for p in ce]
gp=SamplingParams(n=1,temperature=0.0,max_tokens=2048,stop=["```\\n","\\n\\n\\n"])
co=ev.llm.generate(cp,gp)
am=[compute_all_metrics(extract_code(o.outputs[0].text)) for o in co if extract_code(o.outputs[0].text).strip()]
def sm(k):
 v=[m[k] for m in am if k in m];return sum(v)/len(v) if v else 0
for k in ["avg_func_length","magic_numbers","nesting_depth"]:r[k]=sm(k)
r["dead_code_pct"]=sm("dead_code")*100;r["naming_length"]=sm("naming_length");r["label"]="{label}"
json.dump(r,open("{rp}","w"),indent=2)
ev.cleanup();del ev;gc.collect();torch.cuda.empty_cache()
'''
        ef = f"/tmp/ev_{label}.py"
        open(ef,"w").write(sc)
        t0 = time.time()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(GPU), "VLLM_ATTENTION_BACKEND": "FLASH_ATTN"}
        proc = subprocess.run([sys.executable, ef], capture_output=True, text=True, timeout=1800, env=env)
        el = time.time()-t0
        os.remove(ef)
        if proc.returncode != 0:
            print(f"  ERROR: {proc.stderr[-300:]}", flush=True)
        elif os.path.exists(rp):
            r = json.load(open(rp))
            print(f"  Pass@1={r['pass_at_1']:.1%} Pass@8={r['pass_at_k']:.1%} Pylint={r['pylint']:.1f} Cmplx={r['complexity']:.1f} ({el:.0f}s)", flush=True)
    finally:
        if os.path.exists(local): shutil.rmtree(local)
        free = shutil.disk_usage("/").free/(1024**3)
        print(f"  Disk: {free:.0f} GB free", flush=True)

# Summary
print("\n" + "="*110)
hdr = f"{'Label':<22} {'Pass@1':>7} {'Pass@8':>7} {'ECE':>7} {'Pylint':>7} {'Cmplx':>6} {'Cmt%':>6} {'Dup%':>6} {'FuncL':>6} {'Magic':>6} {'Nest':>5}"
print(hdr); print("-"*110)
order = ["base"]+[j[0] for j in JOBS]
for lb in order:
    rp = os.path.join(EVAL_DIR, f"{lb}_results.json")
    if not os.path.exists(rp): continue
    r = json.load(open(rp))
    print(f"{lb:<22} {r['pass_at_1']:>6.1%} {r['pass_at_k']:>6.1%} {r['ece']:>7.4f} {r['pylint']:>7.1f} "
          f"{r['complexity']:>6.1f} {r.get('comment_pct',0):>5.1f}% {r.get('duplication_pct',0):>5.1f}% "
          f"{r.get('avg_func_length',0):>6.1f} {r.get('magic_numbers',0):>6.1f} {r.get('nesting_depth',0):>5.1f}")
print("="*110)
all_r = {}
for lb in order:
    rp = os.path.join(EVAL_DIR, f"{lb}_results.json")
    if os.path.exists(rp): all_r[lb] = json.load(open(rp))
json.dump(all_r, open(os.path.join(EVAL_DIR,"all_v2_results.json"),"w"), indent=2)
