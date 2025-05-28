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

    # Get all unique usernames from both DataFrames
    usernames = set(edits["username"].unique())

    user_rows = []
    for user in usernames:
        user_rows.append({"username": user})

    users_df = pd.DataFrame(user_rows)
    data["users"] = users_df


def add_basic_statistics_to_users(data: Dict[str, pd.DataFrame]) -> None:
    """
    Add basic statistics to the users DataFrame.
    """

    users_df = data["users"]
    edits_df = data["edits"]
    messages_df = data["messages"]
    executions_df = data["executions"]
    interactions_df = data["interactions"]

    users_df["num_edits"] = (
        users_df["username"]
        .map(edits_df["username"].value_counts())
        .fillna(0)
        .astype(int)
    )

    users_df["num_executions"] = (
        users_df["username"]
        .map(executions_df["username"].value_counts())
        .fillna(0)
        .astype(int)
    )

    users_df["num_interactions"] = (
        users_df["username"]
        .map(interactions_df["username"].value_counts())
        .fillna(0)
        .astype(int)
    )

    # Calculate percentage_successful using the 'success' column in executions
    success_counts = executions_df.groupby("username")["success"].sum()
    total_counts = executions_df.groupby("username").size()
    users_df["percentage_successful"] = (
        users_df["username"].map(success_counts)
        / users_df["username"].map(total_counts)
    ).fillna(0)

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
            executions_df[
                ["id", "username", "datetime", "is_previous_execution_success"]
            ],
            left_on="execution_id",
            right_on="id",
            how="left",
        )

        # For each user and each learning goal, create a Series of (datetime, result)
        for goal in learning_goals:
            col_name = f"{goal.name} series"

            def build_result_series(user):
                user_success = success_merged[
                    (success_merged["username"] == user)
                    & (
                        success_merged["learning_goals_of_added_code"].apply(
                            lambda goals: (goal in goals)
                        )
                    )
                ]
                # Filter errors where the user has successfully executed before
                user_error = error_merged[
                    (error_merged["username"] == user)
                    & (error_merged["is_previous_execution_success"] == True)
                    & (
                        error_merged["learning_goals_in_error_by_user_fix"].apply(
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
            col_name = f"{goal.name} average success"

            def compute_average(user_result_df):
                num_true = (user_result_df["result"] == True).sum()
                num_false = (user_result_df["result"] == False).sum()
                total = num_true + num_false
                if total == 0:
                    return None
                return num_true / total

            # Find the column with the result series for this goal
            result_col = f"{goal.name} series"
            users_df[col_name] = users_df[result_col].apply(compute_average)

        data["users"] = users_df

    return add_average_learning_goals_success


def add_bayesian_knowledge_tracing(learning_goals: list[LearningGoal]):
    def add_bkt(data: Dict[str, pd.DataFrame]) -> None:
        """
        For each user and each learning goal, apply Bayesian Knowledge Tracing (BKT) to their result series.
        Adds a new column for each learning goal with a DataFrame of (datetime, p_known).
        """
        users_df = data["users"]

        # BKT parameters (can be tuned)
        p_init = 0.2
        p_learn = 0.2
        p_guess = 0.2
        p_slip = 0.1

        for goal in learning_goals:
            col_name = f"{goal.name} series"
            bkt_col = f"{goal.name} BKT"

            def bkt_trace(result_df):
                p_know = p_init
                index = []
                values = []
                for _, row in result_df.iterrows():
                    correct = row["result"]
                    dt = row["datetime"]
                    if correct:
                        num = p_know * (1 - p_slip)
                        denom = p_know * (1 - p_slip) + (1 - p_know) * p_guess
                    else:
                        num = p_know * p_slip
                        denom = p_know * p_slip + (1 - p_know) * (1 - p_guess)
                    if denom == 0:
                        p_know_given_obs = p_know
                    else:
                        p_know_given_obs = num / denom
                    p_know = p_know_given_obs + (1 - p_know_given_obs) * p_learn
                    index.append(dt)
                    values.append(p_know)

                return pd.DataFrame({"datetime": index, "p_known": values})

            users_df[bkt_col] = users_df[col_name].map(bkt_trace).astype(object)
            users_df[bkt_col] = users_df[bkt_col]

        data["users"] = users_df

    return add_bkt


def add_moving_average(learning_goals: list[LearningGoal], window_size: int):
    def add_moving_average(data: Dict[str, pd.DataFrame]) -> None:
        """
        For each user and each learning goal, compute the moving average of the series
        """
        users_df = data["users"]

        for goal in learning_goals:
            series_col = f"{goal.name} series"
            moving_avg_col = f"{goal.name} moving average"

            def compute_moving_average(series_df):
                if series_df.empty:
                    return pd.DataFrame(columns=["datetime", "moving_average"])
                result_df = series_df.copy()
                result_df["moving_average"] = (
                    result_df["result"]
                    .rolling(window=window_size, min_periods=1)
                    .mean()
                )
                return result_df[["datetime", "moving_average"]]

            users_df[moving_avg_col] = users_df[series_col].map(compute_moving_average)
            users_df[moving_avg_col] = users_df[moving_avg_col].astype(object)

        data["users"] = users_df

    return add_moving_average
