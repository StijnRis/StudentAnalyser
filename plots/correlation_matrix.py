import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_correlation_matrix(dataframe_name: str, columns: list[str], output_dir: str):
    def plot_correlation_matrix(data: dict[str, pd.DataFrame]) -> None:
        # Get the DataFrame for the specified dataframe_name
        df = data[dataframe_name]

        # Convert pd.NA to np.nan and ensure numeric dtype
        df_numeric = df[columns].apply(pd.to_numeric, errors="coerce")
        corr = df_numeric.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title(f"Correlation Matrix for {dataframe_name}")
        plt.tight_layout()

        output_path = f"{output_dir}/correlation_matrix_{dataframe_name}_{'_'.join(columns)}.png"
        plt.savefig(output_path)
        plt.close()

    return plot_correlation_matrix
