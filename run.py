"""
run.py — Energy Grid Load Balancer — Main Entry Point

Usage:
  python run.py                    # full training + evaluation
  python run.py --quick            # 100 episodes, fast test
  python run.py --episodes 500     # custom episode count
  python run.py --eval-only        # evaluate pre-trained (if checkpoint exists)
  python run.py --no-plots         # skip plot generation
"""
import os, sys, argparse, time, json
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

from environments.grid_env import (
    GRID_ENV_REGISTRY, GRID_TRAIN_ENVS, GRID_EVAL_ENVS, GRID_UNSEEN_ENVS
)
from agents.grid_acpl_agent import GridACPLAgent
from agents.baselines        import RuleBasedAgent, RandomAgent
from training.train_grid     import (
    train_agent, evaluate_agent, evaluate_all, print_results_table
)
from evaluation.plots        import generate_all_plots


def _fmt_eta(sec):
    if sec < 0: return "?"
    h, r = divmod(int(sec), 3600); m, s = divmod(r, 60)
    if h: return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def build_agents(seed: int) -> dict:
    return {
        "GridACPL":    GridACPLAgent(seed=seed),
        "RuleBase":    RuleBasedAgent(seed=seed),
        "RandomAgent": RandomAgent(seed=seed),
    }


def collect_lambda_data(agent, n_episodes=20, max_steps=96, seed=88888):
    """Collect (load, stress, lambda) triples for the heatmap."""
    from environments.grid_env import EnergyGridEnv
    from training.train_grid   import run_episode
    pairs = []
    for ep in range(n_episodes):
        env = EnergyGridEnv(max_steps=max_steps, seed=seed+ep)
        agent.reset_hidden()
        state = env.reset()
        while not env.done:
            load   = float(state[0])
            stress = float(state[3])
            s_norm = agent.normalizer.normalize(state)
            lam    = float(agent.lambda_net.forward(s_norm)) * agent.lambda_scale
            pairs.append((load, stress, lam))
            action = agent.select_action(state, eval_mode=True)
            state, _, _, done, _ = env.step(action)
            if done: break
    return pairs


def run_benchmark(args):
    os.makedirs(args.out,     exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print("\n" + "="*70)
    print("  ENERGY GRID LOAD BALANCER — ACPL v6")
    print("="*70)
    print(f"  Episodes:  {args.episodes}")
    print(f"  Eval eps:  {args.eval_episodes}")
    print(f"  Seed:      {args.seed}")
    print(f"  Results →  {args.out}/")
    print(f"  Logs    →  {args.log_dir}/")
    print("="*70 + "\n")

    t_total = time.time()

    # ── Phase 1: Training ─────────────────────────────────────────────────────
    print("── Phase 1: Training ──────────────────────────────────────────────")
    agents    = build_agents(args.seed)
    histories = {}

    for name, agent in agents.items():
        if name in ("RuleBase", "RandomAgent"):
            # Baselines don't train — just create empty history shell
            histories[name] = {"rewards":[], "consequences":[], "delayed_hits":[],
                                "losses":[], "mean_lambda":[], "hit_freq_ema":[],
                                "equipment_stress":[], "frequency":[]}
            print(f"  {name}: no training (rule-based/random)")
            continue

        print(f"\n  Training {name}...")
        t0 = time.time()
        histories[name] = train_agent(
            agent,
            n_episodes  = args.episodes,
            max_steps   = args.max_steps,
            delay_steps = args.delay,
            log_freq    = args.log_freq,
            seed        = args.seed,
            verbose     = args.verbose,
            env_names   = GRID_TRAIN_ENVS,
            log_dir     = args.log_dir,
            save_every  = args.save_every,
        )
        print(f"  ✓ {name} trained in {_fmt_eta(time.time()-t0)}")

    # ── Phase 2: Evaluation ────────────────────────────────────────────────────
    print("\n── Phase 2: Evaluation ────────────────────────────────────────────")
    all_eval_envs = list(GRID_EVAL_ENVS) + list(GRID_UNSEEN_ENVS)
    t0 = time.time()

    results = evaluate_all(
        agents, all_eval_envs,
        n_episodes  = args.eval_episodes,
        max_steps   = args.max_steps,
        delay_steps = args.delay,
        log_dir     = args.log_dir,
    )
    print(f"  ✓ Evaluation done in {_fmt_eta(time.time()-t0)}")
    print_results_table(results, all_eval_envs)

    # ── Phase 3: Lambda analysis ───────────────────────────────────────────────
    print("\n── Phase 3: Lambda risk map ───────────────────────────────────────")
    acpl_agent = agents.get("GridACPL")
    lam_pairs  = []
    delay_hist = []

    if acpl_agent:
        t0 = time.time()
        lam_pairs  = collect_lambda_data(acpl_agent, n_episodes=30,
                                          max_steps=args.max_steps, seed=77777)
        delay_hist = list(acpl_agent._delay_log)
        print(f"  ✓ Lambda data collected ({len(lam_pairs)} points) in {_fmt_eta(time.time()-t0)}")

        # Theory checks
        if lam_pairs:
            lam_vals = [x[2] for x in lam_pairs]
            print(f"\n  Lambda statistics:")
            print(f"    Min λ: {min(lam_vals):.4f}  Max λ: {max(lam_vals):.4f}")
            print(f"    Mean λ: {np.mean(lam_vals):.4f}  Std λ: {np.std(lam_vals):.4f}")
            # Check: lambda higher near overload (load > 0.85)
            high_load = [x[2] for x in lam_pairs if x[0] > 0.85]
            low_load  = [x[2] for x in lam_pairs if x[0] < 0.50]
            if high_load and low_load:
                print(f"    λ near overload (load>85%): {np.mean(high_load):.4f}")
                print(f"    λ at low load   (load<50%): {np.mean(low_load):.4f}")
                print(f"    ✓ State-conditioning {'working' if np.mean(high_load) > np.mean(low_load) else 'needs more training'}")

    # ── Phase 4: Plots ─────────────────────────────────────────────────────────
    if not args.no_plots:
        print("\n── Phase 4: Generating plots ──────────────────────────────────────")
        t0 = time.time()
        generate_all_plots(
            histories       = {k: v for k, v in histories.items() if v.get("rewards")},
            results         = results,
            env_names       = all_eval_envs,
            out_dir         = args.out,
            state_lambda_pairs = lam_pairs,
            delay_history   = delay_hist,
        )

        print(f"  ✓ Plots saved to {args.out}/ in {_fmt_eta(time.time()-t0)}")

    # ── Phase 5: Dashboard ─────────────────────────────────────────────────────
    print("\n── Phase 5: Generating HTML dashboard ────────────────────────────")
    try:
        from dashboard.generate_dashboard import generate_dashboard
        dash_path = os.path.join(args.out, "dashboard.html")
        generate_dashboard(args.log_dir, dash_path, open_browser=False)
        print(f"  ✓ Open in browser: {os.path.abspath(dash_path)}")
    except Exception as e:
        print(f"  Dashboard generation note: {e}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  BENCHMARK SUMMARY")
    print("="*70)
    total_elapsed = time.time() - t_total

    acpl_r = np.mean([results.get("GridACPL", {}).get(e, {}).get("mean_reward", 0)
                      for e in GRID_EVAL_ENVS])
    rule_r = np.mean([results.get("RuleBase",  {}).get(e, {}).get("mean_reward", 0)
                      for e in GRID_EVAL_ENVS])
    acpl_c = np.mean([results.get("GridACPL", {}).get(e, {}).get("mean_consequence", 0)
                      for e in GRID_EVAL_ENVS])
    rule_c = np.mean([results.get("RuleBase",  {}).get(e, {}).get("mean_consequence", 0)
                      for e in GRID_EVAL_ENVS])
    acpl_csr = np.mean([results.get("GridACPL",{}).get(e, {}).get("csr", 0)
                        for e in GRID_EVAL_ENVS])

    print(f"  GridACPL   Reward: {acpl_r:.3f}  J_c: {acpl_c:.4f}  CSR: {acpl_csr:.1f}%")
    print(f"  RuleBase   Reward: {rule_r:.3f}  J_c: {rule_c:.4f}")
    if acpl_r > rule_r:
        print(f"  ✓ ACPL outperforms rule-based by {acpl_r - rule_r:.3f} reward")
    else:
        print(f"  → Rule-based leads by {rule_r - acpl_r:.3f} (try more training episodes)")
    if acpl_c < rule_c:
        print(f"  ✓ ACPL has lower consequence: {acpl_c:.4f} vs {rule_c:.4f}")

    print(f"\n  Total runtime: {_fmt_eta(total_elapsed)}")
    print(f"  Logs:    {args.log_dir}/")
    print(f"  Plots:   {args.out}/")
    print("="*70 + "\n")

    return {"agents": agents, "histories": histories, "results": results}


def main():
    p = argparse.ArgumentParser(description="Energy Grid Load Balancer — ACPL v6")
    p.add_argument("--episodes",      type=int, default=300,
                   help="Training episodes (default 300; use 1000+ for full convergence)")
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--max-steps",     type=int, default=96,    help="Steps per episode (96=1 day)")
    p.add_argument("--delay",         type=int, default=16,    help="Consequence delay steps (16=4h)")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--log-freq",      type=int, default=25)
    p.add_argument("--save-every",    type=int, default=50)
    p.add_argument("--out",           type=str, default="results")
    p.add_argument("--log-dir",       type=str, default="logs")
    p.add_argument("--quick",         action="store_true", help="100 episodes, fast test")
    p.add_argument("--no-plots",      action="store_true")
    p.add_argument("--quiet",         action="store_true")
    args = p.parse_args()

    args.verbose = not args.quiet
    if args.quick:
        args.episodes      = 100
        args.eval_episodes = 20
        args.log_freq      = 10

    run_benchmark(args)


if __name__ == "__main__":
    main()
