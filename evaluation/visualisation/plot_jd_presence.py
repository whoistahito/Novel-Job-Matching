import json
from pathlib import Path
import matplotlib.pyplot as plt
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

contains_jd = [d['containsJobDescription'] for d in all_data]
jd_present = sum(contains_jd)
jd_absent = len(contains_jd) - jd_present
output_dir = Path(__file__).parent


fig, ax = plt.subplots(figsize=(8, 6))
categories = ['Contains Job Description', 'No Job Description']
counts = [jd_present, jd_absent]
colors = ['#90EE90', '#FFB6C6']

bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=1)

ax.set_title('Job Description Presence', fontweight='bold', pad=20)
ax.set_ylabel('Count')

# Add counts and percentages on top of bars
total = sum(counts)
for bar in bars:
    height = bar.get_height()
    percentage = (height / total) * 100
    ax.text(bar.get_x() + bar.get_width()/2., height + (total * 0.01),
            f'{height}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add some padding to top of y-axis to fit labels
ax.set_ylim(0, total * 1.15)

plt.tight_layout()
plt.savefig(output_dir / 'job_description_presence.png', dpi=300, bbox_inches='tight')
print("plot saved")
plt.close()

