import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_scatter_plot(dataframe_name, x, y, output_dir):
    def plot_scatter_plot(data: dict[str, pd.DataFrame]) -> None:
        """
        Generate a scatter plot between columns x and y from the specified dataframe, with a trend line.
        """
        df = data[dataframe_name]
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=x, y=y)
        sns.regplot(
            data=df,
            x=x,
            y=y,
            scatter=False,
            color="red",
            line_kws={"label": "Trend line"},
        )
        plt.title(f"Scatter plot of {y} vs {x}")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{dataframe_name}_{x}_vs_{y}_scatter.png")
        plt.close()

    return plot_scatter_plot
