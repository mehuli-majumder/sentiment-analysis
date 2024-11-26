import matplotlib.pyplot as plt
import numpy as np

# Data for accuracy comparison
models = ["LinearSVC + TF-IDF", "LinearSVC + W2V", "RF + TF-IDF", "RF + W2V", "GBM + TF-IDF", "GBM + W2V"]
accuracy_70_30 = [0.8675, 0.66, 0.77, 0.65, 0.72, 0.64]
accuracy_80_20 = [0.8807, 0.65, 0.78, 0.66, 0.72, 0.64]

# Bar chart for accuracy comparison
x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, accuracy_70_30, width, label='70-30 Split')
rects2 = ax.bar(x + width/2, accuracy_80_20, width, label='80-20 Split')

# Add labels, title, and legend
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()

# Add value annotations
def annotate_bars(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

annotate_bars(rects1)
annotate_bars(rects2)

plt.tight_layout()
plt.show()

# Precision, recall, and F1-score for one model as an example (e.g., LinearSVC + TF-IDF, 80-20)
labels = ["Negative", "Neutral", "Positive"]
precision = [0.91, 0.64, 0.87]
recall = [0.92, 0.64, 0.86]
f1_score = [0.91, 0.64, 0.86]

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1_score, width, label='F1-Score')

# Add labels, title, and legend
ax.set_ylabel('Scores')
ax.set_title('Performance Metrics (LinearSVC + TF-IDF, 80-20 Split)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add value annotations
annotate_bars(rects1)
annotate_bars(rects2)
annotate_bars(rects3)

plt.tight_layout()
plt.show()
