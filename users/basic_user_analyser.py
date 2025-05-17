from typing import Dict

import pandas as pd

from interactions.interaction_analyser import LearningGoal


def add_users_dataframe(data: Dict[str, pd.DataFrame]) -> None:
    """
    Add a DataFrame containing user data to the main data dictionary.
    Each row corresponds to a user found in either the edits or messages DataFrame,
    and includes the number of messages and edits related to that user.
    """

    edits = data["edits"]
    interactions = data["interactions"]

    # Get all unique usernames from both DataFrames
    users_edits = set(edits["username"].unique())
    users_interactions = set(interactions["username"].unique())
    all_users = users_edits.union(users_interactions)

    user_rows = []
    for user in all_users:
        num_interactions = interactions[interactions["username"] == user].shape[0]
        num_edits = edits[edits["username"] == user].shape[0]
        user_rows.append(
            {
                "username": user,
                "num_interactions": num_interactions,
                "num_edits": num_edits,
            }
        )

    users_df = pd.DataFrame(user_rows)
    data["users"] = users_df


def add_learning_goals_result_series(learning_goals: list[LearningGoal]):
    """
    For each user, for each learning goal, create a pandas Series with datetime and result (true for success, false for error).
    """

    def add_learning_goals_result_series(
        data: Dict[str, pd.DataFrame],
    ) -> None:

        users_df = data["users"]
        executions_df = data["executions"]
        execution_successes_df = data["execution_successes"]
        execution_errors_df = data["execution_errors"]

        # Merge execution_successes with executions to get username, datetime, learning_goals_of_added_code
        success_merged = execution_successes_df.merge(
            executions_df[["id", "username", "datetime"]],
            left_on="execution_id",
            right_on="id",
            how="left",
        )
        # Merge execution_errors with executions to get username, datetime, learning_goal_in_error_ai
        error_merged = execution_errors_df.merge(
            executions_df[["id", "username", "datetime"]],
            left_on="execution_id",
            right_on="id",
            how="left",
        )

        # For each user and each learning goal, create a Series of (datetime, result)
        for goal in learning_goals:
            col_name = f"learning_goal_result_series_{goal.name}"

            def build_result_series(user):
                user_success = success_merged[
                    (success_merged["username"] == user)
                    & (
                        success_merged["learning_goals_of_added_code"].apply(
                            lambda goals: (goal in goals)
                        )
                    )
                ]
                user_error = error_merged[
                    (error_merged["username"] == user)
                    & (
                        error_merged["learning_goals_in_error_ai_detection"].apply(
                            lambda goals: (goal in goals)
                        )
                    )
                ]
                success_df = pd.DataFrame(
                    {"datetime": user_success["datetime"], "result": True}
                )
                error_df = pd.DataFrame(
                    {"datetime": user_error["datetime"], "result": False}
                )
                combined = (
                    pd.concat([success_df, error_df])
                    .sort_values("datetime")
                    .reset_index(drop=True)
                )
                return combined

            users_df[col_name] = users_df["username"].map(build_result_series)
            users_df[col_name] = users_df[col_name].astype(object)
        data["users"] = users_df

    return add_learning_goals_result_series


def add_average_learning_goals_success(learning_goals: list[LearningGoal]):
    def add_average_learning_goals_success(data: Dict[str, pd.DataFrame]) -> None:
        """
        For each user, for each learning goal, compute the average success rate.
        """

        users_df = data["users"]
        for goal in learning_goals:
            col_name = f"learning_goal_average_success_{goal.name}"

            def compute_average(user_result_df):
                num_true = (user_result_df["result"] == True).sum()
                num_false = (user_result_df["result"] == False).sum()
                total = num_true + num_false
                if total == 0:
                    return None
                return num_true / total

            # Find the column with the result series for this goal
            result_col = f"learning_goal_result_series_{goal.name}"
            users_df[col_name] = users_df[result_col].apply(compute_average)
        data["users"] = users_df

    return add_average_learning_goals_success
