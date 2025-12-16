import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DEFAULT_DATA = """{
  "evaluated_models": [
    "llama3.1-8b-instruct",
    "mistral-7B-instruct",
    "qwen3-8b",
    "glm4-9b"
  ],
  "metrics_comparison": {
    "Readability [GEval]": {
      "llama3.1-8b-instruct": {
        "average_score": 0.4942583732057416,
        "passed_rate": 0.4019138755980861
      },
      "mistral-7B-instruct": {
        "average_score": 0.5516746411483253,
        "passed_rate": 0.4449760765550239
      },
      "qwen3-8b": {
        "average_score": 0.6066985645933014,
        "passed_rate": 0.5263157894736842
      },
      "glm4-9b": {
        "average_score": 0.5842105263157894,
        "passed_rate": 0.5167464114832536
      }
    },
    "Completeness [GEval]": {
      "llama3.1-8b-instruct": {
        "average_score": 0.36889952153110045,
        "passed_rate": 0.22488038277511962
      },
      "mistral-7B-instruct": {
        "average_score": 0.4043062200956938,
        "passed_rate": 0.18660287081339713
      },
      "qwen3-8b": {
        "average_score": 0.4511961722488038,
        "passed_rate": 0.27751196172248804
      },
      "glm4-9b": {
        "average_score": 0.4789473684210526,
        "passed_rate": 0.36363636363636365
      }
    },
    "Correctness [GEval]": {
      "llama3.1-8b-instruct": {
        "average_score": 0.38229665071770336,
        "passed_rate": 0.23444976076555024
      },
      "mistral-7B-instruct": {
        "average_score": 0.4363636363636364,
        "passed_rate": 0.19138755980861244
      },
      "qwen3-8b": {
        "average_score": 0.44545454545454544,
        "passed_rate": 0.23923444976076555
      },
      "glm4-9b": {
        "average_score": 0.4287081339712918,
        "passed_rate": 0.24401913875598086
      }
    },
    "Alignment [GEval]": {
      "llama3.1-8b-instruct": {
        "average_score": 0.392822966507177,
        "passed_rate": 0.24401913875598086
      },
      "mistral-7B-instruct": {
        "average_score": 0.45885167464114834,
        "passed_rate": 0.2966507177033493
      },
      "qwen3-8b": {
        "average_score": 0.49617224880382776,
        "passed_rate": 0.3492822966507177
      },
      "glm4-9b": {
        "average_score": 0.42727272727272725,
        "passed_rate": 0.27751196172248804
      }
    }
  },
  "rankings": {
    "Readability [GEval]": [
      {
        "rank": 1,
        "model_id": "qwen3-8b",
        "score": 0.6066985645933014
      },
      {
        "rank": 2,
        "model_id": "glm4-9b",
        "score": 0.5842105263157894
      },
      {
        "rank": 3,
        "model_id": "mistral-7B-instruct",
        "score": 0.5516746411483253
      },
      {
        "rank": 4,
        "model_id": "llama3.1-8b-instruct",
        "score": 0.4942583732057416
      }
    ],
    "Completeness [GEval]": [
      {
        "rank": 1,
        "model_id": "glm4-9b",
        "score": 0.4789473684210526
      },
      {
        "rank": 2,
        "model_id": "qwen3-8b",
        "score": 0.4511961722488038
      },
      {
        "rank": 3,
        "model_id": "mistral-7B-instruct",
        "score": 0.4043062200956938
      },
      {
        "rank": 4,
        "model_id": "llama3.1-8b-instruct",
        "score": 0.36889952153110045
      }
    ],
    "Correctness [GEval]": [
      {
        "rank": 1,
        "model_id": "qwen3-8b",
        "score": 0.44545454545454544
      },
      {
        "rank": 2,
        "model_id": "mistral-7B-instruct",
        "score": 0.4363636363636364
      },
      {
        "rank": 3,
        "model_id": "glm4-9b",
        "score": 0.4287081339712918
      },
      {
        "rank": 4,
        "model_id": "llama3.1-8b-instruct",
        "score": 0.38229665071770336
      }
    ],
    "Alignment [GEval]": [
      {
        "rank": 1,
        "model_id": "qwen3-8b",
        "score": 0.49617224880382776
      },
      {
        "rank": 2,
        "model_id": "mistral-7B-instruct",
        "score": 0.45885167464114834
      },
      {
        "rank": 3,
        "model_id": "glm4-9b",
        "score": 0.42727272727272725
      },
      {
        "rank": 4,
        "model_id": "llama3.1-8b-instruct",
        "score": 0.392822966507177
      }
    ],
    "overall": [
      {
        "rank": 1,
        "model_id": "qwen3-8b",
        "average_score": 0.4998803827751196
      },
      {
        "rank": 2,
        "model_id": "glm4-9b",
        "average_score": 0.47978468899521526
      },
      {
        "rank": 3,
        "model_id": "mistral-7B-instruct",
        "average_score": 0.462799043062201
      },
      {
        "rank": 4,
        "model_id": "llama3.1-8b-instruct",
        "average_score": 0.4095693779904306
      }
    ]
  }
}
"""
data = json.loads(DEFAULT_DATA)

records = []
for metric, models_data in data['metrics_comparison'].items():
    for model, scores in models_data.items():
        records.append({
            'model': model,
            'metric': metric.replace(" [GEval]", ""),
            'average_score': scores['average_score'],
            'passed_rate': scores['passed_rate']
        })

df_metrics = pd.DataFrame(records)
df_overall = pd.DataFrame(data['rankings']['overall'])

# --- 2. Visualization using Seaborn ---
sns.set_theme(style="whitegrid", palette="viridis")


# --- Plot 1: Grouped Bar Chart for Average Scores ---
g = sns.catplot(
    data=df_metrics,
    kind="bar",
    x="model",
    y="average_score",
    hue="metric",
    height=7,
    aspect=1.8,
    legend_out=True
)

# Set the numerical axis limit to 1.0 for context
g.set(ylim=(0, 1))

g.set_axis_labels("Model", "Average Score")
g.set_xticklabels(rotation=45, ha='right')
g.legend.set_title("Evaluation Metric")

plt.savefig('average_scores_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# --- Plot 2: Grouped Bar Chart for Passed Rates ---
g = sns.catplot(
    data=df_metrics,
    kind="bar",
    x="model",
    y="passed_rate",
    hue="metric",
    height=7,
    aspect=1.8,
    legend_out=True
)

# Set the numerical axis limit to 1.0 for context
g.set(ylim=(0, 1))

g.set_axis_labels("Model", "Passed Rate")
g.set_xticklabels(rotation=45, ha='right')
g.legend.set_title("Evaluation Metric")

plt.savefig('passed_rates_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# --- Plot 3: Horizontal Bar Chart for Overall Ranking ---
df_overall_sorted = df_overall.sort_values(by='average_score', ascending=False)

plt.figure(figsize=(12, 7))

barplot = sns.barplot(
    x='average_score',
    y='model_id',
    data=df_overall_sorted,
    palette='viridis_r'
)

for index, value in enumerate(df_overall_sorted['average_score']):
    plt.text(value + 0.01, index, f'{value:.3f}', va='center', fontsize=11, color='black')

# Set the numerical axis limit to 1.0 for context
plt.xlim(0, 1)

plt.xlabel('Overall Average Score (out of 1.0)', fontsize=12)
plt.ylabel('Model', fontsize=12)

plt.tight_layout()
plt.savefig('overall_model_ranking.png', dpi=300, bbox_inches='tight')
plt.show()