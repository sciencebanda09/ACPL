import subprocess
import sys
import time

SEEDS    = [42, 123, 777, 999, 2024]   # 5 seeds
EPISODES = 20000

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"  Starting seed {seed}")
    print(f"{'='*60}\n")
    
    t0  = time.time()
    cmd = [
        sys.executable, "run.py",
        "--episodes",  str(EPISODES),
        "--seed",      str(seed),
        "--out",       f"results_seed{seed}",
        "--log-dir",   f"logs_seed{seed}",
        "--quiet",
    ]
    subprocess.run(cmd, check=True)
    elapsed = (time.time() - t0) / 3600
    print(f"\n  Seed {seed} done in {elapsed:.1f}h")

print("\nAll seeds complete.")