import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_violin_plot(dataframe_name: str, x: str, y: str, output_dir: str):
    def plot_violin_plot(data: dict[str, pd.DataFrame]) -> None:
        """
        Generate a violin plot between columns x and y from the specified dataframe.
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
            elif isinstance(point2, pd.Timedelta):
                records.append({x: str(point1), y: point2.total_seconds()})
            elif isinstance(point2, (float, int)):
                records.append({x: str(point1), y: point2})
            else:
                print(f"Skipping row with unexpected types: {point1}, {point2}")
        plot_df = pd.DataFrame(records)

        # Drop missing values
        plot_df = plot_df.dropna(subset=[x, y])

        # Count occurrences for each x value
        x_counts = plot_df[x].value_counts().sort_index()
        # Create a mapping from x value to label with count
        x_label_map = {val: f"{val} (n={count})" for val, count in x_counts.items()}
        # Map the x column to the new labels
        plot_df["x_label_with_count"] = plot_df[x].map(x_label_map)

        # Plot
        plt.figure(figsize=(19.20, 10.80), dpi=100)
        sns.violinplot(x="x_label_with_count", y=y, data=plot_df)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{x} vs {y}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/violin_plot_{dataframe_name}_{x}_vs_{y}.png")
        plt.close()

    return plot_violin_plot
