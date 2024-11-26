import numpy as np
import matplotlib.pyplot as plt

# Data for confusion matrix analysis
models = ["LinearSVC + TF-IDF", "LinearSVC + W2V", "RF + TF-IDF", "RF + W2V", "GBM + TF-IDF", "GBM + W2V"]
classes = ["Negative", "Neutral", "Positive"]

# Updated confusion matrix counts based on user-provided data
true_positives = [
    [1709, 92, 1067],  # LinearSVC + TF-IDF
    [1563, 1, 586],    # LinearSVC + W2V
    [1600, 14, 898],   # RF + TF-IDF
    [1602, 14, 524],   # RF + W2V
    [1767, 8, 573],    # GBM + TF-IDF
    [1619, 10, 471],   # GBM + W2V
]
false_positives = [
    [34, 42, 191],
    [2, 33, 339],
    [3, 34, 301],
    [1, 19, 301],
    [3, 23, 134],
    [0, 19, 285],
]
false_negatives = [
    [161, 59, 184],
    [339, 112, 672],
    [301, 65, 365],
    [301, 80, 739],
    [134, 82, 689],
    [285, 84, 790],
]

# Plot stacked bar chart
x = np.arange(len(models))
width = 0.2

for i, class_name in enumerate(classes):
    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width, [tp[i] for tp in true_positives], width, label='True Positives', color='g')
    rects2 = ax.bar(x, [fp[i] for fp in false_positives], width, label='False Positives', color='r')
    rects3 = ax.bar(x + width, [fn[i] for fn in false_negatives], width, label='False Negatives', color='b')

    # Add labels, title, and legend
    ax.set_ylabel('Counts')
    ax.set_title(f'Confusion Matrix Analysis ({class_name} Class)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()

    # Annotate bars
    def annotate_bars(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    annotate_bars(rects1)
    annotate_bars(rects2)
    annotate_bars(rects3)

    plt.tight_layout()
    plt.show()
