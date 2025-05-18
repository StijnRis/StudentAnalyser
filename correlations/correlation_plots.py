import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_violin_plot(dataframe_name, x, y, output_dir):
    def plot_violin_plot(data: dict[str, pd.DataFrame]) -> None:
        """
        Generate a violin plot mapping each learning goal in question_learning_goals to increase_in_success_rate.
        Each interaction's increase_in_success_rate is counted for every learning goal in its list.
        """
        dataframe = data[dataframe_name]

        # Build a DataFrame with one row per (learning_goal, increase_in_success_rate)
        records = []
        for _, row in dataframe.iterrows():
            point1 = row[x]
            point2 = row[y]
            if isinstance(point1, list):
                for item in point1:
                    records.append({x: str(item), y: float(point2)})
            else:
                records.append({x: str(point1), y: float(point2)})
        plot_df = pd.DataFrame(records)

        # Drop missing values
        plot_df = plot_df.dropna(subset=[x, y])

        # Plot
        plt.figure(figsize=(19.20, 10.80), dpi=100)
        sns.violinplot(x=x, y=y, data=plot_df)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{x} vs {y}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{x}_vs_{y}_violin_plot.png")
        plt.close()

    return plot_violin_plot
