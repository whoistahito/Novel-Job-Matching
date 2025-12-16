import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 15

data_dir = Path(__file__).parent / "job_description_characteristics"
all_data = []

for json_file in data_dir.glob("*.json"):
    if json_file.name == "analysis_summary.json":
        continue
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.append(data)
    except Exception as e:
        print(f"Error loading {json_file}: {e}")

character_counts = [d['characterCount'] for d in all_data]
output_dir = Path(__file__).parent

fig, ax = plt.subplots(figsize=(12, 7))
n, bins, patches = ax.hist(character_counts, bins=30, edgecolor='black',
                             alpha=0.75, color='steelblue', linewidth=1.5)

cm = plt.cm.viridis
norm = plt.Normalize(vmin=bins.min(), vmax=bins.max())
for i, patch in enumerate(patches):
    patch.set_facecolor(cm(norm(bins[i])))

ax.set_xlabel('Character Count', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Distribution of Job Description Lengths', fontweight='bold', pad=20)

mean_val = np.mean(character_counts)
median_val = np.median(character_counts)
ax.axvline(mean_val, color='red', linestyle='--',
            label=f'Mean: {mean_val:.0f}', linewidth=2.5)
ax.axvline(median_val, color='green', linestyle='--',
            label=f'Median: {median_val:.0f}', linewidth=2.5)

stats_text = f'Total: {len(character_counts)}\n'
stats_text += f'Std Dev: {np.std(character_counts):.0f}\n'
stats_text += f'Min: {np.min(character_counts):.0f}\n'
stats_text += f'Max: {np.max(character_counts):.0f}'
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.legend(loc='upper right', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / 'distribution_job_description_lengths.png', dpi=300, bbox_inches='tight')
print("plot saved")
plt.close()

