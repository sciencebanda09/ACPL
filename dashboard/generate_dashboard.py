"""
dashboard/generate_dashboard.py

Generates a self-contained HTML dashboard from training logs.
No server needed — open the HTML file directly in any browser.

Usage:
  python dashboard/generate_dashboard.py --log-dir logs/ --out results/dashboard.html
  python dashboard/generate_dashboard.py --log-dir logs/  # auto-opens in browser
"""
import os, sys, json, argparse, webbrowser
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def load_training_log(log_dir: str, agent_name: str = "GridACPL") -> dict:
    path = os.path.join(log_dir, f"{agent_name}_training.json")
    if not os.path.exists(path):
        # Try underscore variant
        for fname in os.listdir(log_dir):
            if fname.endswith("_training.json"):
                path = os.path.join(log_dir, fname)
                break
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_eval_results(log_dir: str) -> dict:
    path = os.path.join(log_dir, "eval_results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def smooth(arr, w=10):
    if len(arr) < w: return arr
    kernel = [1/w]*w
    result = []
    for i in range(len(arr)):
        start = max(0, i-w+1)
        result.append(sum(arr[start:i+1]) / (i-start+1))
    return result


def build_dashboard_html(history: dict, eval_results: dict) -> str:
    """Build a complete self-contained HTML dashboard."""

    # Prepare data series
    episodes  = list(range(len(history.get("rewards", []))))
    rewards   = history.get("rewards", [])
    conseqs   = history.get("consequences", [])
    freqs     = history.get("frequency", [])
    stresses  = history.get("equipment_stress", [])
    lambdas   = history.get("mean_lambda", [])
    hit_ema   = history.get("hit_freq_ema", [])
    lam_scale = history.get("lambda_scale", [])
    exp_delay = history.get("expected_delay", [])
    cum_r     = history.get("cumulative_reward", [])
    battery   = history.get("battery_soc", [])
    shed      = history.get("load_shed_mw", [])

    sr  = smooth(rewards, 20)
    sc  = smooth(conseqs, 20)
    sf  = smooth(freqs, 10)
    ss  = smooth(stresses, 20)
    sl  = smooth(lambdas, 10)

    # Eval table rows
    eval_rows = ""
    for agent_name, envs in eval_results.items():
        for env_name, metrics in envs.items():
            r   = metrics.get("mean_reward", 0)
            c   = metrics.get("mean_consequence", 0)
            csr = metrics.get("csr", 0)
            dh  = metrics.get("mean_delayed_hits", 0)
            fr  = metrics.get("mean_frequency", 50)
            st  = metrics.get("mean_stress", 0)
            ms  = metrics.get("mean_infer_ms", 0)
            csr_cls = "good" if csr > 70 else ("warn" if csr > 40 else "bad")
            eval_rows += f"""
            <tr>
              <td>{agent_name}</td><td>{env_name}</td>
              <td class="num">{r:.3f}</td>
              <td class="num">{c:.4f}</td>
              <td class="num {csr_cls}">{csr:.1f}%</td>
              <td class="num">{dh:.2f}</td>
              <td class="num">{fr:.3f}</td>
              <td class="num">{st:.4f}</td>
              <td class="num">{ms:.3f}</td>
            </tr>"""

    # Stats summary
    n = len(rewards)
    final_r  = float(np.mean(rewards[-50:])) if n > 50 else (float(np.mean(rewards)) if rewards else 0)
    final_c  = float(np.mean(conseqs[-50:])) if n > 50 else (float(np.mean(conseqs)) if conseqs else 0)
    final_f  = float(np.mean(freqs[-50:]))   if n > 50 else 50.0
    final_s  = float(np.mean(stresses[-50:]))if n > 50 else 0.0
    final_l  = float(np.mean(lambdas[-20:])) if len(lambdas) > 20 else 0.0
    final_ht = float(hit_ema[-1]) if hit_ema else 0.0

    def js_arr(lst): return "[" + ",".join(f"{v:.4f}" for v in lst) + "]"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Energy Grid — ACPL Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f1117; --bg2: #1a1d27; --bg3: #232736;
    --border: #2e3350; --text: #e2e8f0; --text2: #94a3b8;
    --blue: #3b82f6; --green: #22c55e; --amber: #f59e0b;
    --red: #ef4444; --purple: #a855f7; --teal: #14b8a6;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif; font-size: 14px; }}
  header {{ background: var(--bg2); border-bottom: 1px solid var(--border); padding: 16px 24px; display: flex; align-items: center; gap: 16px; }}
  header h1 {{ font-size: 18px; font-weight: 600; color: var(--text); }}
  header .badge {{ background: var(--blue); color: #fff; font-size: 11px; padding: 2px 8px; border-radius: 12px; }}
  .subtitle {{ color: var(--text2); font-size: 12px; }}
  .main {{ padding: 20px 24px; max-width: 1600px; margin: 0 auto; }}

  /* KPI cards */
  .kpi-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-bottom: 24px; }}
  .kpi {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }}
  .kpi .label {{ font-size: 11px; color: var(--text2); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
  .kpi .value {{ font-size: 24px; font-weight: 700; }}
  .kpi .sub {{ font-size: 11px; color: var(--text2); margin-top: 4px; }}
  .kpi.good .value {{ color: var(--green); }}
  .kpi.warn .value {{ color: var(--amber); }}
  .kpi.bad  .value {{ color: var(--red); }}
  .kpi.neutral .value {{ color: var(--blue); }}
  .kpi.purple .value {{ color: var(--purple); }}

  /* Charts */
  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }}
  .chart-grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 24px; }}
  .chart-card {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }}
  .chart-card h3 {{ font-size: 13px; font-weight: 600; color: var(--text2); margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .chart-card canvas {{ max-height: 220px; }}

  /* Table */
  .table-card {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-bottom: 24px; overflow-x: auto; }}
  .table-card h2 {{ font-size: 14px; font-weight: 600; margin-bottom: 14px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: var(--bg3); color: var(--text2); font-weight: 500; padding: 8px 12px; text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid var(--border); }}
  td.num {{ text-align: right; font-family: monospace; }}
  td.good {{ color: var(--green); font-weight: 600; }}
  td.warn {{ color: var(--amber); font-weight: 600; }}
  td.bad  {{ color: var(--red);  font-weight: 600; }}
  tr:hover td {{ background: var(--bg3); }}

  /* Section headings */
  .section-title {{ font-size: 15px; font-weight: 600; color: var(--text); margin: 24px 0 12px; padding-left: 4px; border-left: 3px solid var(--blue); padding-left: 10px; }}

  /* Frequency gauge */
  .freq-bar {{ height: 10px; border-radius: 5px; background: var(--bg3); margin-top: 8px; position: relative; overflow: hidden; }}
  .freq-fill {{ height: 100%; border-radius: 5px; transition: width 0.5s; }}

  /* Footer */
  footer {{ text-align: center; padding: 20px; color: var(--text2); font-size: 11px; border-top: 1px solid var(--border); margin-top: 20px; }}
</style>
</head>
<body>

<header>
  <div>
    <h1>⚡ Energy Grid Load Balancer</h1>
    <div class="subtitle">ACPL v6 — Adaptive Consequence-Penalized Learning</div>
  </div>
  <span class="badge">ACPL v6</span>
  <span class="badge" style="background:var(--green)">Training Complete</span>
  <div style="margin-left:auto;color:var(--text2);font-size:12px;">{n} episodes trained</div>
</header>

<div class="main">

  <!-- KPI Cards -->
  <div class="section-title">Final Performance (last 50 episodes)</div>
  <div class="kpi-grid">
    <div class="kpi {'good' if final_r > -50 else 'warn' if final_r > -100 else 'bad'}">
      <div class="label">Mean Reward</div>
      <div class="value">{final_r:.1f}</div>
      <div class="sub">episode avg</div>
    </div>
    <div class="kpi {'good' if final_c < 50 else 'warn' if final_c < 100 else 'bad'}">
      <div class="label">Consequence J_c</div>
      <div class="value">{final_c:.2f}</div>
      <div class="sub">lower = safer</div>
    </div>
    <div class="kpi {'good' if abs(final_f-50) < 0.2 else 'warn' if abs(final_f-50) < 0.5 else 'bad'}">
      <div class="label">Grid Frequency</div>
      <div class="value">{final_f:.3f}</div>
      <div class="sub">Hz (target: 50.0)</div>
    </div>
    <div class="kpi {'good' if final_s < 0.1 else 'warn' if final_s < 0.3 else 'bad'}">
      <div class="label">Equipment Stress</div>
      <div class="value">{final_s:.4f}</div>
      <div class="sub">0=none, 1=critical</div>
    </div>
    <div class="kpi purple">
      <div class="label">Mean λ(s)</div>
      <div class="value">{final_l:.3f}</div>
      <div class="sub">consequence weight</div>
    </div>
    <div class="kpi {'good' if final_ht < 0.2 else 'warn' if final_ht < 0.5 else 'bad'}">
      <div class="label">Hit Frequency</div>
      <div class="value">{final_ht:.2f}</div>
      <div class="sub">EMA rate</div>
    </div>
  </div>

  <!-- Training charts row 1 -->
  <div class="section-title">Training Dynamics</div>
  <div class="chart-grid">
    <div class="chart-card">
      <h3>Episode Reward</h3>
      <canvas id="rewardChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Consequence J_c (Delayed)</h3>
      <canvas id="conseqChart"></canvas>
    </div>
  </div>

  <div class="chart-grid-3">
    <div class="chart-card">
      <h3>Grid Frequency (Hz)</h3>
      <canvas id="freqChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Equipment Stress</h3>
      <canvas id="stressChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Cumulative Reward</h3>
      <canvas id="cumChart"></canvas>
    </div>
  </div>

  <!-- Lambda & delay row -->
  <div class="section-title">ACPL Internals — Lambda & Delay Estimator</div>
  <div class="chart-grid">
    <div class="chart-card">
      <h3>State-Conditioned λ(s) + Warmup Scale</h3>
      <canvas id="lambdaChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Hit Frequency EMA + Expected Delay E[τ|h]</h3>
      <canvas id="delayChart"></canvas>
    </div>
  </div>

  <!-- Grid operations row -->
  <div class="chart-grid">
    <div class="chart-card">
      <h3>Battery State of Charge</h3>
      <canvas id="batteryChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Load Shed (MW)</h3>
      <canvas id="shedChart"></canvas>
    </div>
  </div>

  <!-- Evaluation table -->
  <div class="section-title">Evaluation Results — All Environments</div>
  <div class="table-card">
    <h2>Benchmark Table</h2>
    <table>
      <thead>
        <tr>
          <th>Agent</th><th>Environment</th>
          <th>Reward ↑</th><th>J_c ↓</th><th>CSR% ↑</th>
          <th>Hits</th><th>Freq (Hz)</th><th>Stress ↓</th><th>ms/step</th>
        </tr>
      </thead>
      <tbody>{eval_rows if eval_rows else '<tr><td colspan="9" style="text-align:center;color:var(--text2)">No evaluation results yet — run with --eval-episodes > 0</td></tr>'}</tbody>
    </table>
  </div>

  <!-- ACPL info box -->
  <div class="section-title">How ACPL Works on This Grid</div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px;">
    <div class="chart-card">
      <h3>🔁 GRU Temporal Memory</h3>
      <p style="color:var(--text2);font-size:13px;line-height:1.6;margin-top:8px;">
        The GRU hidden state captures 24-hour price and demand cycles.
        When prices peak at 19:00, the agent has already seen the morning ramp
        and adjusts generator dispatch before the peak arrives.
      </p>
    </div>
    <div class="chart-card">
      <h3>⚡ State Lambda λ(s)</h3>
      <p style="color:var(--text2);font-size:13px;line-height:1.6;margin-top:8px;">
        The lambda network raises the consequence penalty near capacity limits
        (load > 85%) and high equipment stress. In safe operating regions,
        lambda stays low — allowing normal economic dispatch without over-constraining.
      </p>
    </div>
    <div class="chart-card">
      <h3>⏱ Delay Estimator P(τ|h)</h3>
      <p style="color:var(--text2);font-size:13px;line-height:1.6;margin-top:8px;">
        Equipment stress signals arrive ~4 hours (16 steps) after overload events.
        The delay estimator learns this distribution from experience, allowing
        the agent to attribute billing consequences to the correct earlier actions.
      </p>
    </div>
  </div>

</div>

<footer>
  Energy Grid Load Balancer · ACPL v6 · Generated {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}
</footer>

<script>
const eps = {js_arr(episodes)};
const N   = eps.length;

// Downsample for performance
function ds(arr, maxPts=400) {{
  if (arr.length <= maxPts) return arr;
  const step = Math.ceil(arr.length / maxPts);
  return arr.filter((_, i) => i % step === 0);
}}
function dsEps(arr, maxPts=400) {{
  if (arr.length <= maxPts) return arr;
  const step = Math.ceil(arr.length / maxPts);
  return arr.filter((_, i) => i % step === 0);
}}

const rawR  = {js_arr(rewards)};
const smR   = {js_arr(sr)};
const rawC  = {js_arr(conseqs)};
const smC   = {js_arr(sc)};
const rawF  = {js_arr(freqs)};
const smF   = {js_arr(sf)};
const rawS  = {js_arr(stresses)};
const smS   = {js_arr(ss)};
const rawL  = {js_arr(lambdas)};
const smL   = {js_arr(sl)};
const rawHE = {js_arr(hit_ema)};
const rawLS = {js_arr(lam_scale)};
const rawED = {js_arr(exp_delay)};
const rawCR = {js_arr(cum_r)};
const rawBA = {js_arr(battery)};
const rawSH = {js_arr(shed)};

const defOpts = (title, yLabel) => ({{
  responsive: true, maintainAspectRatio: true,
  animation: {{ duration: 300 }},
  plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }}, tooltip: {{ mode: 'index' }} }},
  scales: {{
    x: {{ ticks: {{ color: '#64748b', maxTicksLimit: 6 }}, grid: {{ color: '#1e2335' }} }},
    y: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e2335' }}, title: {{ display: !!yLabel, text: yLabel, color: '#64748b' }} }}
  }}
}});

function mkChart(id, datasets, yLabel='', extra={{}}) {{
  const ctx = document.getElementById(id);
  if (!ctx) return;
  const epsDS = dsEps([...Array(N).keys()]);
  new Chart(ctx, {{ type: 'line', data: {{
    labels: epsDS,
    datasets: datasets.map(d => ({{ ...d, data: ds(d.rawData), tension: 0.3, pointRadius: 0, borderWidth: d.borderWidth||1.5 }}))
  }}, options: {{ ...defOpts('', yLabel), ...extra }} }});
}}

mkChart('rewardChart', [
  {{ rawData: rawR, label: 'Raw',      borderColor: '#3b82f620', fill: false }},
  {{ rawData: smR,  label: 'Smoothed', borderColor: '#3b82f6', borderWidth: 2, fill: false }},
]);
mkChart('conseqChart', [
  {{ rawData: rawC, label: 'Raw',      borderColor: '#ef444420', fill: false }},
  {{ rawData: smC,  label: 'Smoothed', borderColor: '#ef4444', borderWidth: 2, fill: false }},
]);
mkChart('freqChart', [
  {{ rawData: rawF, label: 'Frequency (Hz)', borderColor: '#22c55e', fill: false }},
], 'Hz', {{
  scales: {{ y: {{ min: 49, max: 51, ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e2335' }} }},
             x: {{ ticks: {{ color: '#64748b', maxTicksLimit: 6 }}, grid: {{ color: '#1e2335' }} }} }}
}});
mkChart('stressChart', [
  {{ rawData: rawS,  label: 'Raw',      borderColor: '#f59e0b30', fill: false }},
  {{ rawData: smS,   label: 'Smoothed', borderColor: '#f59e0b', borderWidth: 2, fill: false }},
]);
mkChart('cumChart', [
  {{ rawData: rawCR, label: 'Cumulative Reward', borderColor: '#14b8a6', borderWidth: 2, fill: true,
     backgroundColor: '#14b8a610' }},
]);
mkChart('lambdaChart', [
  {{ rawData: smL,   label: 'λ(s) smoothed',  borderColor: '#a855f7', borderWidth: 2, fill: false }},
  {{ rawData: rawLS, label: 'Warmup scale',    borderColor: '#64748b', borderWidth: 1, fill: false }},
]);
mkChart('delayChart', [
  {{ rawData: rawHE, label: 'Hit freq EMA',     borderColor: '#ef4444', borderWidth: 2, fill: false }},
  {{ rawData: rawED, label: 'E[τ|h] (steps)',   borderColor: '#3b82f6', borderWidth: 2, fill: false }},
]);
mkChart('batteryChart', [
  {{ rawData: rawBA, label: 'Battery SoC', borderColor: '#22c55e', borderWidth: 1.5, fill: true,
     backgroundColor: '#22c55e15' }},
], '0-1');
mkChart('shedChart', [
  {{ rawData: rawSH, label: 'Load Shed MW', borderColor: '#f59e0b', borderWidth: 1.5, fill: true,
     backgroundColor: '#f59e0b15' }},
], 'MW');
</script>
</body>
</html>"""
    return html


def generate_dashboard(log_dir: str, out_path: str, open_browser: bool = True):
    print(f"  Loading logs from {log_dir}/...")
    history      = load_training_log(log_dir)
    eval_results = load_eval_results(log_dir)

    if not history:
        print(f"  Warning: No training log found in {log_dir}/")
        print(f"  Run 'python run.py' first to generate training data.")
        history = {"rewards": [], "consequences": [], "frequency": [],
                   "equipment_stress": [], "mean_lambda": [], "hit_freq_ema": [],
                   "lambda_scale": [], "expected_delay": [], "cumulative_reward": [],
                   "battery_soc": [], "load_shed_mw": []}

    html = build_dashboard_html(history, eval_results)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  Dashboard saved → {out_path}")
    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(out_path)}")
        print(f"  Opening in browser...")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate ACPL Grid Dashboard")
    p.add_argument("--log-dir",  type=str, default="logs")
    p.add_argument("--out",      type=str, default="results/dashboard.html")
    p.add_argument("--no-open",  action="store_true", help="Don't open browser")
    args = p.parse_args()
    generate_dashboard(args.log_dir, args.out, open_browser=not args.no_open)
