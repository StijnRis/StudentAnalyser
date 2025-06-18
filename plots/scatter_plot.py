import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns


def _get_global_axis_limits(df, x_columns, y_column):
    x_min = float("inf")
    x_max = float("-inf")
    y_min = float("inf")
    y_max = float("-inf")
    for col in x_columns:
        col_min = pd.to_numeric(df[col], errors="coerce").min()
        col_max = pd.to_numeric(df[col], errors="coerce").max()
        if pd.notna(col_min):
            x_min = min(x_min, col_min)
        if pd.notna(col_max):
            x_max = max(x_max, col_max)
    y_col_min = pd.to_numeric(df[y_column], errors="coerce").min()
    y_col_max = pd.to_numeric(df[y_column], errors="coerce").max()
    if pd.notna(y_col_min):
        y_min = min(y_min, y_col_min)
    if pd.notna(y_col_max):
        y_max = max(y_max, y_col_max)
    # Always include (0,0)
    x_min = min(x_min, 0)
    x_max = max(x_max, 0)
    y_min = min(y_min, 0)
    y_max = max(y_max, 0)
    return x_min, x_max, y_min, y_max


def _plot_scatter_with_stats(ax, x, y, label=None, color=None):
    ax.scatter(x, y, label=label, color=color)
    sns.regplot(
        x=x,
        y=y,
        scatter=False,
        color=color or "red",
        line_kws={"alpha": 0.7},
        ax=ax,
    )
    if len(x) > 1:
        corr_coef, p_value = stats.pearsonr(x.values, y.values)
        stats_text = f"r = {corr_coef:.2f}\np = {p_value:.2g}"
    else:
        stats_text = "Not enough data"
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )


def plot_scatter_plot(
    dataframe_name: str, group_by: str, x: str, y: str, output_dir: str
):
    def plot_scatter_plot(data: dict[str, pd.DataFrame]) -> None:
        df = data[dataframe_name]
        df = df[[x, y, group_by]].copy()
        df[x] = pd.to_numeric(df[x], errors="coerce")
        df[y] = pd.to_numeric(df[y], errors="coerce")
        df = df.dropna(subset=[x, y, group_by])
        if group_by not in df.columns:
            raise ValueError(
                f"DataFrame must contain a '{group_by}' column for splitting."
            )
        x_min, x_max, y_min, y_max = _get_global_axis_limits(df, [x], y)
        for value in df[group_by].unique():
            subset = df[df[group_by] == value]
            fig, ax = plt.subplots(figsize=(8, 6))
            _plot_scatter_with_stats(ax, subset[x], subset[y])
            ax.set_title(f"Scatter plot of {y} vs {x} for {group_by} = {value}")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            fig.tight_layout()
            fig.savefig(
                f"{output_dir}/{dataframe_name}_{group_by}_{value}_{x}_vs_{y}_scatter.png"
            )
            plt.close(fig)

    return plot_scatter_plot


def plot_scatter_plot_with_multiple_datasets(
    dataframe_name: str,
    group_by: str,
    x_label: str,
    x_columns: list[str],
    y_column: str,
    output_dir: str,
):
    def plot_scatter_plot_with_multiple_datasets(data: dict[str, pd.DataFrame]) -> None:
        df = data[dataframe_name]
        if group_by not in df.columns:
            raise ValueError(
                f"DataFrame must contain a '{group_by}' column for splitting."
            )
        x_min, x_max, y_min, y_max = _get_global_axis_limits(df, x_columns, y_column)
        group_values = df[group_by].dropna().unique()
        for group_value in group_values:
            group_df = df[df[group_by] == group_value]
            n_items = len(x_columns)
            ncols = min(4, n_items)
            nrows = (n_items + ncols - 1) // ncols
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False
            )
            colors = sns.color_palette("colorblind", n_items)
            for i, col in enumerate(x_columns):
                ax = axes[i // ncols, i % ncols]
                x = pd.to_numeric(group_df[col], errors="coerce")
                y = pd.to_numeric(group_df[y_column], errors="coerce")
                valid = ~(x.isna() | y.isna())
                x = x[valid]
                y = y[valid]
                if len(x) == 0:
                    ax.set_visible(False)
                    continue
                _plot_scatter_with_stats(
                    ax,
                    x,
                    y,
                    label=col.replace("num_", "").replace("_questions", ""),
                    color=colors[i],
                )
                ax.set_xlabel(col)
                ax.set_ylabel(y_column)
                ax.set_title(f"{y_column} vs. {col}")
                ax.legend()
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
            for idx in range(n_items, nrows * ncols):
                fig.delaxes(axes[idx // ncols, idx % ncols])
            plt.tight_layout()
            plt.savefig(
                f"{output_dir}/{dataframe_name}_{group_by}_{group_value}_{x_label}_vs_{y_column}_scatter_grid.png"
            )
            plt.close(fig)

    return plot_scatter_plot_with_multiple_datasets
