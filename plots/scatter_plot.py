import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_scatter_plot(dataframe_name: str, x: str, y: str, output_dir: str):
    def plot_scatter_plot(data: dict[str, pd.DataFrame]) -> None:
        """
        Generate a scatter plot between columns x and y from the specified dataframe, with a trend line.
        """
        df = data[dataframe_name]

        # Drop rows where x or y is missing or not numeric
        df = df[[x, y]].copy()
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna()

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
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{dataframe_name}_{x}_vs_{y}_scatter.png")
        plt.close()

    return plot_scatter_plot


def plot_scatter_plot_with_mutliple_lines(
    output_dir: str,
    dataframe_name: str,
    x_label: str,
    x_columns: list[str],
    y_column: str,
):
    """
    Plot the amount of interaction types against the grade.
    """

    def plot(data: dict[str, pd.DataFrame]) -> None:
        df = data[dataframe_name]

        plt.figure(figsize=(10, 7))
        colors = sns.color_palette("colorblind", len(x_columns))

        for i, col in enumerate(x_columns):
            # Ensure numeric and drop NaNs
            x = pd.to_numeric(df[col], errors="coerce")
            y = pd.to_numeric(df[y_column], errors="coerce")
            valid = ~(x.isna() | y.isna())
            x = x[valid]
            y = y[valid]
            if len(x) == 0:
                continue
            sns.scatterplot(
                x=x,
                y=y,
                label=col.replace("num_", "").replace("_questions", ""),
                color=colors[i],
            )
            # Plot trend line
            sns.regplot(
                x=x,
                y=y,
                scatter=False,
                color=colors[i],
                line_kws={"alpha": 0.7},
            )
        plt.xlabel(x_label)
        plt.ylabel(y_column)
        plt.title(f"{y_column} vs. {x_label}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{x_label}_vs_{y_column}_scatter.png")
        plt.close()

    return plot
