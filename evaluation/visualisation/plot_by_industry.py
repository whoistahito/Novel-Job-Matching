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

industries = [d['industry'] for d in all_data]
industry_counts = Counter(industries)
output_dir = Path(__file__).parent

fig, ax = plt.subplots(figsize=(10, 6))
industries_sorted = sorted(industry_counts.items(), key=lambda x: x[1], reverse=True)
ind_names, ind_counts = zip(*industries_sorted)
colors = plt.cm.Set3(np.linspace(0, 1, len(ind_names)))
bars = ax.barh(ind_names, ind_counts, color=colors, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Number of Job Descriptions', fontweight='bold')
ax.set_ylabel('Industry', fontweight='bold')
ax.set_title('Job Descriptions by Industry', fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3, linestyle='--')

for i, (bar, count) in enumerate(zip(bars, ind_counts)):
    ax.text(count + 1, i, str(count), va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'job_descriptions_by_industry.png', dpi=300, bbox_inches='tight')
print("plot saved")
plt.close()

