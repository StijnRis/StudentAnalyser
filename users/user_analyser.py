from typing import Dict

import pandas as pd

from enums import QuestionType
from interactions.interaction_analyser import LearningGoal


def add_basic_execution_statistics(data: Dict[str, pd.DataFrame]) -> None:
    """
    Add basic statistics to the users DataFrame.
    """

    users_df = data["users"]
    edits_df = data["edits"]
    messages_df = data["messages"]
    executions_df = data["executions"]
    interactions_df = data["interactions"]

    users_df["num_edits"] = (
        users_df["user_id"]
        .map(edits_df["user_id"].value_counts())
        .fillna(0)
        .astype(int)
    )

    users_df["num_executions"] = (
        users_df["user_id"]
        .map(executions_df["user_id"].value_counts())
        .fillna(0)
        .astype(int)
    )

    # Calculate percentage of executions that were successful
    success_counts = executions_df.groupby("user_id")["success"].sum()
    total_counts = executions_df.groupby("user_id").size()
    users_df["execution_success_rate"] = (
        users_df["user_id"].map(success_counts) / users_df["user_id"].map(total_counts)
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

        # Merge execution_successes with executions to get user_id, datetime, learning_goals_of_added_code
        success_merged = execution_successes_df.merge(
            executions_df[["execution_id", "user_id", "datetime"]],
            left_on="execution_id",
            right_on="execution_id",
            how="left",
        )
        # Merge execution_errors with executions to get user_id, datetime, learning_goal_in_error_ai
        error_merged = execution_errors_df.merge(
            executions_df[
                ["execution_id", "user_id", "datetime", "is_previous_execution_success"]
            ],
            left_on="execution_id",
            right_on="execution_id",
            how="left",
        )

        # For each user and each learning goal, create a Series of (datetime, result)
        for goal in learning_goals:
            col_name = f"{goal.name}_series"

            def build_result_series(user):
                user_success = success_merged[
                    (success_merged["user_id"] == user)
                    & (
                        success_merged["learning_goals_of_added_code"].apply(
                            lambda goals: (goal in goals)
                        )
                    )
                ]
                # Filter errors where the user has successfully executed before
                user_error = error_merged[
                    (error_merged["user_id"] == user)
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

            users_df[col_name] = users_df["user_id"].map(build_result_series)
            users_df[col_name] = users_df[col_name].astype(object)
        data["users"] = users_df

    return add_learning_goals_result_series


def add_basic_learning_goals_statistics(learning_goals: list[LearningGoal]):
    def add_basic_learning_goals_statistics(data: Dict[str, pd.DataFrame]) -> None:
        """
        For each user, for each learning goal, compute the average success rate and the slope coefficient.
        """
        import numpy as np
        from scipy.stats import linregress

        users_df = data["users"]

        for goal in learning_goals:
            col_name = f"{goal.name} average success"
            slope_col = f"{goal.name} slope"

            def compute_average(user_result_df):
                num_true = (user_result_df["result"] == True).sum()
                num_false = (user_result_df["result"] == False).sum()
                total = num_true + num_false
                if total == 0:
                    return None
                return num_true / total

            def compute_slope(user_result_df):
                if user_result_df.empty or len(user_result_df) < 2:
                    return None
                x = user_result_df["datetime"].map(lambda dt: dt.timestamp()).values
                y = user_result_df["result"].astype(float).values

                slope, intercept, r, p, se = linregress(x, y)
                return slope

            # Find the column with the result series for this goal
            result_col = f"{goal.name}_series"
            users_df[col_name] = users_df[result_col].apply(compute_average)
            users_df[slope_col] = users_df[result_col].apply(compute_slope)

        data["users"] = users_df

    return add_basic_learning_goals_statistics


def add_bayesian_knowledge_tracing(learning_goals: list[LearningGoal]):
    def add_bkt(data: Dict[str, pd.DataFrame]) -> None:
        """
        For each user and each learning goal, apply Bayesian Knowledge Tracing (BKT) to their result series.
        Adds a new column for each learning goal with a DataFrame of (datetime, p_known).
        """
        users_df = data["users"]

        # BKT parameters (can be tuned)
        p_init = 0.1
        p_learn = 0.2
        p_guess = 0.2
        p_slip = 0.2

        for goal in learning_goals:
            col_name = f"{goal.name}_series"
            bkt_col = f"{goal.name}_BKT"

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
            series_col = f"{goal.name}_series"
            moving_avg_col = f"{goal.name}_moving_average"

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


def add_basic_interaction_statistics(question_types: list[QuestionType]):
    """
    Add basic interaction statistics to the users DataFrame.
    """
    def add_basic_interaction_statistics(data: Dict[str, pd.DataFrame]):
        users_df = data["users"]
        interactions_df = data["interactions"]

        users_df["num_interactions"] = (
            users_df["user_id"]
            .map(interactions_df["user_id"].value_counts())
            .fillna(0)
            .astype(int)
        )

        # Add columns for each question_type
        for qtype in question_types:
            col_name = f"num_{qtype.name}_questions"
            counts = interactions_df[interactions_df["question_type"] == qtype][
                "user_id"
            ].value_counts()
            users_df[col_name] = users_df["user_id"].map(counts).fillna(0).astype(int)

        data["users"] = users_df
    
    return add_basic_interaction_statistics
