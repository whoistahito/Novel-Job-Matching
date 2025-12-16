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

fig, ax = plt.subplots(figsize=(10, 8))
colors_pie = ['#90EE90', '#FFB6C6']
explode = (0.1, 0)
sizes = [jd_present, jd_absent]
labels = ['Contains Job Description', 'No Job Description']

wedges, texts, autotexts = ax.pie(sizes,
                                     labels=labels,
                                     autopct='%1.2f%%',
                                     startangle=90,
                                     colors=colors_pie,
                                     explode=explode,
                                     shadow=True,
                                     textprops={'fontsize': 12})

ax.set_title('Job Description Presence', fontweight='bold', pad=20)

for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(13)
    autotext.set_weight('bold')

for text in texts:
    text.set_fontsize(13)
    text.set_weight('bold')

legend_labels = [
    f'Contains JD: {jd_present} ({jd_present/len(all_data)*100:.1f}%)',
    f'No JD: {jd_absent} ({jd_absent/len(all_data)*100:.1f}%)'
]
ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1),
           framealpha=0.9, fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'job_description_presence.png', dpi=300, bbox_inches='tight')
print("plot saved")
plt.close()

