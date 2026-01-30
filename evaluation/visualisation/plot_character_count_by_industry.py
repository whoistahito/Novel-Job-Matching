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
industries = [d['industry'] for d in all_data]
output_dir = Path(__file__).parent

fig, ax = plt.subplots(figsize=(14, 8))

unique_industries = sorted(set(industries))
colors_map = plt.cm.tab10(np.linspace(0, 1, len(unique_industries)))
industry_color_dict = {ind: colors_map[i] for i, ind in enumerate(unique_industries)}

for ind in unique_industries:
    indices = [i for i, x in enumerate(industries) if x == ind]
    chars = [character_counts[i] for i in indices]
    ax.scatter(indices, chars, label=ind, alpha=0.7, s=50,
                color=industry_color_dict[ind], edgecolors='black', linewidth=0.5)

ax.set_xlabel('Job Description Index', fontweight='bold')
ax.set_ylabel('Character Count', fontweight='bold')
ax.set_title('Character Count vs. Index (by Industry)', fontweight='bold', pad=20)
ax.legend(title='Industry', bbox_to_anchor=(1.05, 1), loc='upper left',
           framealpha=0.9, title_fontsize=11)
ax.grid(alpha=0.3, linestyle='--')

mean_val = np.mean(character_counts)
median_val = np.median(character_counts)
ax.axhline(mean_val, color='red', linestyle='--', alpha=0.5,
            label=f'Mean: {mean_val:.0f}', linewidth=2)
ax.axhline(median_val, color='green', linestyle='--', alpha=0.5,
            label=f'Median: {median_val:.0f}', linewidth=2)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title='Industry', bbox_to_anchor=(1.05, 1),
           loc='upper left', framealpha=0.9, title_fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'character_count_vs_index_by_industry.png', dpi=300, bbox_inches='tight')
print("plot saved")
plt.close()

