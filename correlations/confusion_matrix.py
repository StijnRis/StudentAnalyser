import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_confusion_matrix(dataframe_name, x, y, output_dir):
    def plot_confusion_matrix(data: dict[str, pd.DataFrame]) -> None:
        df = data[dataframe_name].copy()
        # Convert both columns to string representations
        df[x] = df[x].astype(str)
        df[y] = df[y].astype(str)
        # Get all unique labels from both columns, sort for consistent order
        all_labels = sorted(set(df[x].unique()) | set(df[y].unique()))
        # Generate the confusion matrix with fixed labels/order
        cm = pd.crosstab(df[x], df[y], rownames=[x], colnames=[y], dropna=False)
        cm = cm.reindex(index=all_labels, columns=all_labels, fill_value=0)
        # Plot the confusion matrix
        plt.figure(figsize=(19.20, 10.80), dpi=100)
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
