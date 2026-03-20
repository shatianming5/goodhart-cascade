#!/usr/bin/env python3
"""Filter TACO problems by base model pass@8 rate.

Usage:
  CUDA_VISIBLE_DEVICES=0 python data/filter_by_base_passrate.py --shard_id 0 --num_shards 10
  # After all shards complete:
  python data/filter_by_base_passrate.py --merge --min_pass 0.125 --max_pass 0.875
"""
import argparse
import json
import os
import sys
import glob
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen2.5-Coder-7B-Instruct')
    p.add_argument('--shard_id', type=int, default=0)
    p.add_argument('--num_shards', type=int, default=10)
    p.add_argument('--n_samples', type=int, default=8, help='Number of generations per problem')
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--max_new_tokens', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=8, help='Batch size for generation')
    p.add_argument('--output_dir', default='data/passrate_shards')
    # Merge mode
    p.add_argument('--merge', action='store_true')
    p.add_argument('--min_pass', type=float, default=0.125, help='Min pass@8 rate (1/8)')
    p.add_argument('--max_pass', type=float, default=0.875, help='Max pass@8 rate (7/8)')
    p.add_argument('--min_tests', type=int, default=5)
    return p.parse_args()


def load_all_problems(min_tests=5):
    """Load raw TACO and format into problems with test cases."""
    from goodhart.data.prepare_taco import format_taco_problem
    data_path = os.path.join(os.path.dirname(__file__), 'taco_train.json')
    with open(data_path) as f:
        raw = json.load(f)
    problems = []
    for i, row in enumerate(raw):
        prob = format_taco_problem(row)
        if len(prob['test_cases']) >= min_tests:
            prob['raw_idx'] = i
            prob['difficulty'] = row.get('difficulty', 'UNKNOWN')
            problems.append(prob)
    return problems


def format_prompt(prob):
    """Format problem into chat prompt."""
    text = (
        'Solve the following programming problem in Python.\n\n'
        f'{prob["question"]}\n\n'
    )
    if prob.get('starter_code'):
        text += f'Starter code:\n```python\n{prob["starter_code"]}\n```\n\n'
    text += 'Provide your solution in a Python code block.'
    return [{'role': 'user', 'content': text}]


def run_shard(args):
    """Run pass@8 evaluation on a shard of problems."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from goodhart.rewards.test_passing import extract_code_from_response
    from goodhart.utils.code_exec import run_all_tests

    problems = load_all_problems(args.min_tests)
    total = len(problems)

    # Shard
    shard_size = (total + args.num_shards - 1) // args.num_shards
    start = args.shard_id * shard_size
    end = min(start + shard_size, total)
    shard = problems[start:end]
    print(f'[Shard {args.shard_id}] Processing {len(shard)} problems (idx {start}-{end-1} of {total})')

    # Load model
    try:
        import flash_attn
        attn = 'flash_attention_2'
    except ImportError:
        attn = 'sdpa'
    print(f'Loading {args.model} (attn={attn})...')
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=attn,
        device_map='auto', trust_remote_code=True
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    t0 = time.time()

    for pi, prob in enumerate(shard):
        # Format prompt
        messages = format_prompt(prob)
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Generate n_samples solutions
        inputs = tokenizer([prompt_text], return_tensors='pt', padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        pass_count = 0
        for si in range(0, args.n_samples, args.batch_size):
            bs = min(args.batch_size, args.n_samples - si)
            batch_inputs = {k: v.expand(bs, -1) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **batch_inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = batch_inputs['input_ids'].shape[1]
            for oi in range(bs):
                gen_ids = outputs[oi][prompt_len:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)

                try:
                    code = extract_code_from_response(text)
                    all_pass, _, _ = run_all_tests(code, prob['test_cases'], timeout=3, max_memory_mb=512)
                    if all_pass:
                        pass_count += 1
                except Exception:
                    pass

        pass_rate = pass_count / args.n_samples
        results.append({
            'raw_idx': prob['raw_idx'],
            'difficulty': prob['difficulty'],
            'n_tests': len(prob['test_cases']),
            'pass_count': pass_count,
            'n_samples': args.n_samples,
            'pass_rate': pass_rate,
        })

        elapsed = time.time() - t0
        eta = elapsed / (pi + 1) * (len(shard) - pi - 1)
        print(f'[Shard {args.shard_id}] {pi+1}/{len(shard)} | pass@{args.n_samples}={pass_rate:.3f} '
              f'({pass_count}/{args.n_samples}) | difficulty={prob["difficulty"]} | '
              f'ETA {eta/60:.1f}min')

    # Save shard results
    out_path = os.path.join(args.output_dir, f'shard_{args.shard_id}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[Shard {args.shard_id}] Done! Saved {len(results)} results to {out_path}')
    print(f'[Shard {args.shard_id}] Total time: {(time.time()-t0)/60:.1f}min')

    # Summary
    pass_any = sum(1 for r in results if r['pass_count'] > 0)
    print(f'[Shard {args.shard_id}] Problems with pass@{args.n_samples}>0: {pass_any}/{len(results)}')


def merge_shards(args):
    """Merge all shard results and create filtered dataset."""
    from goodhart.data.prepare_taco import format_taco_problem

    # Load all shards
    shard_files = sorted(glob.glob(os.path.join(args.output_dir, 'shard_*.json')))
    if not shard_files:
        print(f'No shard files found in {args.output_dir}')
        sys.exit(1)

    all_results = []
    for sf in shard_files:
        with open(sf) as f:
            all_results.extend(json.load(f))
    print(f'Loaded {len(all_results)} results from {len(shard_files)} shards')

    # Stats
    from collections import Counter
    pass_dist = Counter()
    diff_stats = {}
    n_samples = all_results[0]['n_samples'] if all_results else 8
    for r in all_results:
        pass_dist[r['pass_count']] += 1
        d = r['difficulty']
        if d not in diff_stats:
            diff_stats[d] = {'total': 0, 'pass_any': 0, 'in_range': 0}
        diff_stats[d]['total'] += 1
        if r['pass_count'] > 0:
            diff_stats[d]['pass_any'] += 1
        if args.min_pass <= r['pass_rate'] <= args.max_pass:
            diff_stats[d]['in_range'] += 1

    print(f'\nPass count distribution (out of {n_samples}):')
    for k in sorted(pass_dist.keys()):
        print(f'  {k} passes: {pass_dist[k]} problems')

    print(f'\nBy difficulty:')
    for d in ['EASY', 'MEDIUM', 'MEDIUM_HARD', 'HARD', 'VERY_HARD', 'UNKNOWN_DIFFICULTY']:
        if d in diff_stats:
            s = diff_stats[d]
            print(f'  {d}: {s["total"]} total, {s["pass_any"]} pass>0, {s["in_range"]} in [{args.min_pass:.3f},{args.max_pass:.3f}]')

    # Filter
    filtered_idx = set()
    for r in all_results:
        if args.min_pass <= r['pass_rate'] <= args.max_pass:
            filtered_idx.add(r['raw_idx'])
    print(f'\nFiltered: {len(filtered_idx)} problems with pass rate in [{args.min_pass:.3f}, {args.max_pass:.3f}]')

    # Load raw data and create TRL format
    data_path = os.path.join(os.path.dirname(__file__), 'taco_train.json')
    with open(data_path) as f:
        raw = json.load(f)

    # Build pass rate lookup
    pass_lookup = {r['raw_idx']: r for r in all_results}

    train_records = []
    for i, row in enumerate(raw):
        if i not in filtered_idx:
            continue
        prob = format_taco_problem(row)
        if len(prob['test_cases']) < args.min_tests:
            continue
        prompt_text = (
            'Solve the following programming problem in Python.\n\n'
            f'{prob["question"]}\n\n'
        )
        if prob.get('starter_code'):
            prompt_text += f'Starter code:\n```python\n{prob["starter_code"]}\n```\n\n'
        prompt_text += 'Provide your solution in a Python code block.'
        train_records.append({
            'prompt': [{'role': 'user', 'content': prompt_text}],
            'ground_truth': json.dumps(prob['test_cases']),
            'pass_rate': pass_lookup[i]['pass_rate'],
            'difficulty': row.get('difficulty', 'UNKNOWN'),
        })

    # Split train/val (90/10)
    import random
    random.seed(42)
    random.shuffle(train_records)
    n_val = max(50, len(train_records) // 10)
    val_records = train_records[:n_val]
    train_records = train_records[n_val:]

    train_path = os.path.join(os.path.dirname(__file__), 'trl_train_filtered.json')
    val_path = os.path.join(os.path.dirname(__file__), 'trl_val_filtered.json')

    with open(train_path, 'w') as f:
        json.dump(train_records, f)
    with open(val_path, 'w') as f:
        json.dump(val_records, f)

    print(f'\nSaved: {len(train_records)} train -> {train_path}')
    print(f'Saved: {len(val_records)} val -> {val_path}')

    # Save full results
    full_path = os.path.join(args.output_dir, 'all_results.json')
    with open(full_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'Full results: {full_path}')


if __name__ == '__main__':
    args = parse_args()
    if args.merge:
        merge_shards(args)
    else:
        run_shard(args)
