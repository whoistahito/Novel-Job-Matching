import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
output_dir = Path(__file__).parent

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

source_data = defaultdict(list)
for d in all_data:
    filename = d.get('filename', '')
    if filename:
        source = filename.split('_')[0]
        if source not in ['indeed', 'linkedin']:
            source = 'Others'
        character_count = d.get('characterCount', 0)
        source_data[source].append(character_count)

source_medians = {source: np.median(counts) for source, counts in source_data.items()}
sorted_sources = sorted(source_medians.keys(), key=lambda x: source_medians[x], reverse=True)
source_labels = {
    "indeed": "Indeed",
    "linkedin": "LinkedIn",
    "Others": "Others",
}
plot_data = []
plot_labels = []
for source in sorted_sources:
    counts = source_data[source]
    plot_data.append(counts)
    label = f"{source_labels.get(source)}\n(n={len(counts)})"
    plot_labels.append(label)

fig, ax = plt.subplots(figsize=(14, 8))

parts = ax.violinplot(plot_data,
                      positions=range(1, len(plot_data) + 1),
                      showmeans=True,
                      showmedians=True,
                      showextrema=True)

for pc in parts['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')
    pc.set_linewidth(1.5)

parts['cmeans'].set_color('red')
parts['cmeans'].set_linewidth(2)
parts['cmedians'].set_color('darkblue')
parts['cmedians'].set_linewidth(2)
parts['cbars'].set_color('black')
parts['cmaxes'].set_color('black')
parts['cmins'].set_color('black')

ax.set_xticks(range(1, len(plot_labels) + 1))
ax.set_xticklabels(plot_labels, rotation=45, ha='right')
ax.set_xlabel('Source', fontsize=13, fontweight='bold')
ax.set_ylabel('Character Count', fontsize=13, fontweight='bold')
ax.set_title('Distribution of Job Posting Character Counts by Source',
             fontsize=16, fontweight='bold', pad=20)

ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

legend_elements = [
    plt.Line2D([0], [0], color='darkblue', linewidth=2, label='Median'),
    plt.Line2D([0], [0], color='red', linewidth=2, label='Mean')
]
ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

textstr = f'Total job postings: {len(all_data)}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()

output_path = output_dir / "character_count_by_source_violin.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print("plot saved")
plt.close()

