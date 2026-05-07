"""Plot throughput vs CPU usage trend across concurrency levels.

Reads benchmark/results/concurrency_*.json and produces a dual-axis figure:
  - left  Y: throughput (msgs/s) bar
  - right Y: backend CPU mean / max (%) lines
  - secondary panel: p50 / p95 / p99 latency vs concurrency
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).parent / 'results'
LEVELS = [50, 100, 500]

data = []
for n in LEVELS:
    d = json.loads((RESULTS / f'concurrency_{n}.json').read_text())
    lat = d['latency_ms']
    data.append({
        'n': n,
        'throughput': d['throughput_msgs_per_s'],
        'cpu_mean': d['backend_cpu_percent_mean'],
        'cpu_max':  d['backend_cpu_percent_max'],
        'rss_max':  d['backend_rss_mb_max'],
        'p50': lat['p50'], 'p95': lat['p95'], 'p99': lat['p99'],
        'mean': lat['mean'],
    })

xs = np.arange(len(LEVELS))
labels = [f'{n}' for n in LEVELS]

fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(13, 4.6))

# ---- Panel 1: Throughput (bar) + CPU (lines) ----
bars = ax1.bar(xs, [d['throughput'] for d in data],
               width=0.55, color='#2563eb', alpha=0.72,
               label='Throughput (msg/s)', zorder=1)
# Throughput labels: place INSIDE the top of each bar in white bold,
# so they cannot collide with the CPU lines/markers drawn on the twin axis.
for b, d in zip(bars, data):
    h = b.get_height()
    # Place label near the BOTTOM of each bar (in white, with a translucent
    # blue plate) so it sits well below any CPU marker drawn on the twin
    # axis and cannot be occluded.
    ax1.text(b.get_x() + b.get_width() / 2, h * 0.10,
             f"{d['throughput']:.1f}", ha='center', va='bottom',
             fontsize=11, color='white', fontweight='bold', zorder=4,
             bbox=dict(boxstyle='round,pad=0.20', fc='#1d4ed8',
                       ec='none', alpha=0.85))

ax1.set_xticks(xs)
ax1.set_xticklabels(labels)
ax1.set_xlim(-0.55, len(LEVELS) - 1 + 0.55)
ax1.set_xlabel('Concurrent WebSocket clients')
ax1.set_ylabel('Throughput  (messages / second)', color='#1e3a8a')
ax1.tick_params(axis='y', labelcolor='#1e3a8a')
ax1.set_ylim(0, max(d['throughput'] for d in data) * 1.30)
ax1.grid(True, axis='y', alpha=0.3, zorder=0)

ax2 = ax1.twinx()
ax2.plot(xs, [d['cpu_mean'] for d in data], 'o-', color='#dc2626',
         linewidth=2.2, markersize=9, label='Backend CPU mean (%)',
         zorder=3)
ax2.plot(xs, [d['cpu_max']  for d in data], 's--', color='#f97316',
         linewidth=2.0, markersize=8, label='Backend CPU peak (%)',
         zorder=3)

# Annotate CPU points with offset boxes so labels never sit on the
# markers / lines and never overlap each other vertically.
mean_bbox = dict(boxstyle='round,pad=0.4', fc='white',
                 ec='#dc2626', lw=0.8, alpha=0.95)
peak_bbox = dict(boxstyle='round,pad=0.4', fc='white',
                 ec='#f97316', lw=0.8, alpha=0.95)
for x, d in zip(xs, data):
    # CPU mean label: below-right of marker
    ax2.annotate(f"{d['cpu_mean']:.1f}%",
                 xy=(x, d['cpu_mean']), xycoords='data',
                 xytext=(10, -14), textcoords='offset points',
                 color='#dc2626', fontsize=9, fontweight='bold',
                 bbox=mean_bbox, zorder=5)
    # CPU peak label: above-right of marker
    ax2.annotate(f"{d['cpu_max']:.1f}%",
                 xy=(x, d['cpu_max']), xycoords='data',
                 xytext=(10, 10), textcoords='offset points',
                 color='#f97316', fontsize=9, fontweight='bold',
                 bbox=peak_bbox, zorder=5)
ax2.set_ylabel('Backend CPU usage  (%)', color='#dc2626')
ax2.tick_params(axis='y', labelcolor='#dc2626')
ax2.set_ylim(0, max(d['cpu_max'] for d in data) * 1.35)

# Combined legend
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=9, framealpha=0.95)
ax1.set_title('(a) Throughput & Backend CPU Usage vs. Concurrency',
              fontsize=11, fontweight='bold', pad=10)

# ---- Panel 2: Latency percentiles ----
ax3.plot(xs, [d['p50']  for d in data], 'o-', color='#0891b2',
         linewidth=2.2, markersize=9, label='p50 (median)')
ax3.plot(xs, [d['p95']  for d in data], 's-', color='#d97706',
         linewidth=2.2, markersize=9, label='p95')
ax3.plot(xs, [d['p99']  for d in data], '^-', color='#dc2626',
         linewidth=2.2, markersize=9, label='p99')
ax3.fill_between(xs,
                 [d['p50'] for d in data],
                 [d['p99'] for d in data],
                 color='#fca5a5', alpha=0.18)
p99_bbox = dict(boxstyle='round,pad=0.4', fc='white',
                ec='#dc2626', lw=0.8, alpha=0.95)
for x, d in zip(xs, data):
    ax3.annotate(f"p99 = {d['p99']:.0f} ms",
                 xy=(x, d['p99']), xycoords='data',
                 xytext=(0, 18), textcoords='offset points',
                 ha='center', color='#dc2626', fontsize=9,
                 fontweight='bold', bbox=p99_bbox, zorder=5)
ax3.set_xticks(xs)
ax3.set_xticklabels(labels)
ax3.set_xlim(-0.55, len(LEVELS) - 1 + 0.55)
ax3.set_xlabel('Concurrent WebSocket clients')
ax3.set_ylabel('End-to-end latency  (ms, log scale)')
ax3.set_yscale('log')
# Provide extra headroom on the log axis so the p99 annotation at x=500
# does not collide with the panel title.
y_top = max(d['p99'] for d in data) * 2.4
y_bot = min(d['p50'] for d in data) * 0.55
ax3.set_ylim(y_bot, y_top)
ax3.grid(True, alpha=0.3, which='both')
ax3.legend(loc='upper left', fontsize=9, framealpha=0.95)
ax3.set_title('(b) WebSocket Push Latency Percentiles vs. Concurrency',
              fontsize=11, fontweight='bold', pad=10)

# ---- Hardware footnote ----
hw = ('Hardware: AMD EPYC 7543 (16 vCPU) | 62 GiB RAM | NVIDIA RTX 4090 24 GB | '
      'PyTorch 2.9.1 + CUDA 13.0 | Linux x86_64')
fig.text(0.5, -0.02, hw, ha='center', fontsize=8.5,
         color='#475569', style='italic')

plt.tight_layout()
out = Path(__file__).parent / 'results' / 'concurrency_trend.png'
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')

out_pdf = out.with_suffix('.pdf')
plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f'Saved: {out_pdf}')
