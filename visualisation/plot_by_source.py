import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
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

sources = [d['filename'].split('_')[0] for d in all_data]
source_counts = Counter(sources)
output_dir = Path(__file__).parent

fig, ax = plt.subplots(figsize=(10, 8))
source_names = list(source_counts.keys())
source_values = list(source_counts.values())
colors_source = plt.cm.Pastel1(np.linspace(0, 1, len(source_names)))

wedges, texts, autotexts = ax.pie(source_values,
                                     labels=source_names,
                                     autopct='%1.1f%%',
                                     startangle=90,
                                     colors=colors_source,
                                     shadow=True,
                                     explode=[0.05] * len(source_names),
                                     textprops={'fontsize': 11})

ax.set_title('Job Descriptions by Source', fontweight='bold', pad=20)

for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(11)
    autotext.set_weight('bold')

for text in texts:
    text.set_fontsize(12)
    text.set_weight('bold')

legend_labels = [f'{name}: {count}' for name, count in zip(source_names, source_values)]
ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1),
           framealpha=0.9, fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'job_descriptions_by_source.png', dpi=300, bbox_inches='tight')
print("plot saved")
plt.close()

