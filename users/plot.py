import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from enums import QuestionType


def plot_amount_of_interaction_types_vs_grade(
    output_dir: str, question_types: list[QuestionType]
):
    """
    Plot the amount of interaction types against the grade.
    """

    def plot(data: dict[str, pd.DataFrame]) -> None:
        users_df = data["users"]
        # Find all columns that match the pattern 'num_*_questions'
        question_cols = [f"num_{qtype.name}_questions" for qtype in question_types]
        plt.figure(figsize=(10, 7))
        colors = sns.color_palette("husl", len(question_cols))
        for i, col in enumerate(question_cols):
            # Only plot if there is at least one nonzero value
            if users_df[col].sum() > 0:
                sns.scatterplot(
                    x=users_df[col],
                    y=users_df["grade"],
                    label=col.replace("num_", "").replace("_questions", ""),
                    color=colors[i],
                )
                # Plot trend line
                sns.regplot(
                    x=users_df[col],
                    y=users_df["grade"],
                    scatter=False,
                    color=colors[i],
                    line_kws={"alpha": 0.7},
                )
        plt.xlabel("Number of questions of specific type")
        plt.ylabel("Grade")
        plt.title("Grade vs. Number of Questions by Type")
        plt.legend(title="Question Type")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/users_question_types_vs_grade_scatter.png")
        plt.close()

    return plot
