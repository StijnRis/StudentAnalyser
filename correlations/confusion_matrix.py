import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_confusion_matrix(dataframe_name, x, y, output_dir):
    def plot_confusion_matrix(data: dict[str, pd.DataFrame]) -> None:
        df = data[dataframe_name].copy()
        # Convert both columns to string representations
        df[x] = df[x].astype(str)
        df[y] = df[y].astype(str)
        # Generate the confusion matrix
        cm = pd.crosstab(df[x], df[y], rownames=[x], colnames=[y], dropna=False)
        # Plot the confusion matrix
        plt.figure(figsize=(max(8, len(cm.columns) * 0.8), max(6, len(cm.index) * 0.6)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix: {x} vs {y}")
        plt.xlabel(y)
        plt.ylabel(x)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{dataframe_name}_{x}_vs_{y}_confusion_matrix.png")
        plt.close()

    return plot_confusion_matrix
