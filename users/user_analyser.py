from typing import Dict

import pandas as pd
from scipy.stats import linregress

from enums import QuestionPurpose, QuestionType
from interactions.interaction_analyser import LearningGoal


def add_basic_user_statistics(data: Dict[str, pd.DataFrame]) -> None:
    """
    Add basic statistics to the users DataFrame.
    """

    users_df = data["users"]
    edits_df = data["edits"]
    executions_df = data["executions"]
    interactions_df = data["interactions"]

    # Calculate number of messages
    users_df["num_edits"] = (
        users_df["user_id"]
        .map(edits_df["user_id"].value_counts())
        .fillna(0)
        .astype(int)
    )

    # Calculate number of executions
    users_df["num_executions"] = (
        users_df["user_id"]
        .map(executions_df["user_id"].value_counts())
        .fillna(0)
        .astype(int)
    )

    # Calculate number of interactions
    users_df["num_interactions"] = (
        users_df["user_id"]
        .map(interactions_df["user_id"].value_counts())
        .fillna(0)
        .astype(int)
    )

    # Calculate percentage of executions that were successful
    success_counts = executions_df.groupby("user_id")["success"].sum()
    total_counts = executions_df.groupby("user_id").size()
    users_df["execution_success_rate"] = (
        users_df["user_id"].map(success_counts) / users_df["user_id"].map(total_counts)
    ).fillna(0)

    # Calculate number of different files (unique file names per user)
    unique_files_per_user = executions_df.groupby("user_id")["filename"].nunique()
    users_df["num_executed_files"] = (
        users_df["user_id"].map(unique_files_per_user).fillna(0).astype(int)
    )

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

        # Collect new columns in a dict
        new_cols = {}

        def build_result_series(user, goal):
            user_success = success_merged[
                (success_merged["user_id"] == user)
                & (
                    success_merged["learning_goals_of_added_code"].apply(
                        lambda goals: goal in goals
                    )
                )
            ]
            user_error = error_merged[
                (error_merged["user_id"] == user)
                & (error_merged["is_previous_execution_success"] == True)
                & (
                    error_merged["learning_goals_in_error_by_user_fix"].apply(
                        lambda goals: goal in goals
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

        for goal in learning_goals:
            col_name = f"{goal.name}_series"
            new_cols[col_name] = (
                users_df["user_id"]
                .map(lambda user: build_result_series(user, goal))
                .astype(object)
            )

        # Assign all new columns at once
        users_df = pd.concat(
            [users_df, pd.DataFrame(new_cols, index=users_df.index)], axis=1
        )
        data["users"] = users_df.copy()

    return add_learning_goals_result_series


def add_construct_result_series(data: Dict[str, pd.DataFrame]) -> None:
    """
    For each user, for each construct, create a pandas Series with datetime and result (true for success, false for error).
    """
    users_df = data["users"]
    executions_df = data["executions"]
    execution_successes_df = data["execution_successes"]
    execution_errors_df = data["execution_errors"]

    # Merge execution_successes with executions to get user_id, datetime, etc.
    success_merged = execution_successes_df.merge(
        executions_df, on="execution_id", how="left"
    )
    error_merged = execution_errors_df.merge(
        executions_df, on="execution_id", how="left"
    )

    # Collect all constructs
    all_constructs = set()
    for constructs in success_merged["added_constructs_as_string"].dropna():
        all_constructs.update(constructs)
    for constructs in error_merged[
        "changed_constructs_as_string_next_success"
    ].dropna():
        all_constructs.update(constructs)

    # Helper function to build result series for a user and construct
    def build_result_series(user, construct):
        user_success = success_merged[
            (success_merged["user_id"] == user)
            & (
                success_merged["added_constructs_as_string"].apply(
                    lambda constructs: construct in constructs
                )
            )
        ]
        user_error = error_merged[
            (error_merged["user_id"] == user)
            & (error_merged["is_previous_execution_success"] == True)
            & (
                error_merged["changed_constructs_as_string_next_success"].apply(
                    lambda constructs: construct in constructs
                )
            )
        ]
        success_df = pd.DataFrame(
            {"datetime": user_success["datetime"], "result": True}
        )
        error_df = pd.DataFrame({"datetime": user_error["datetime"], "result": False})
        combined = (
            pd.concat([success_df, error_df])
            .sort_values("datetime")
            .reset_index(drop=True)
        )
        return combined

    # Build all new columns at once
    new_cols = {}
    for construct in all_constructs:
        col_name = f"{construct}_construct_series"
        new_cols[col_name] = (
            users_df["user_id"]
            .map(lambda user: build_result_series(user, construct))
            .astype(object)
        )

    # Assign all new columns at once
    users_df = pd.concat(
        [users_df, pd.DataFrame(new_cols, index=users_df.index)], axis=1
    )
    data["users"] = users_df.copy()


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
        p_guess = 0.1
        p_slip = 0.1

        new_cols = {}
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

            new_cols[bkt_col] = users_df[col_name].map(bkt_trace).astype(object)
        # Assign all new columns at once to avoid fragmentation
        users_df = pd.concat(
            [users_df, pd.DataFrame(new_cols, index=users_df.index)], axis=1
        )
        data["users"] = users_df

    return add_bkt


def add_moving_average(learning_goals: list[LearningGoal], window_size: int):
    def add_moving_average(data: Dict[str, pd.DataFrame]) -> None:
        """
        For each user and each learning goal, compute the moving average of the series
        """
        users_df = data["users"]
        new_cols = {}
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

            new_col = users_df[series_col].map(compute_moving_average).astype(object)
            new_cols[moving_avg_col] = new_col

        # Concatenate all new columns at once to avoid fragmentation
        users_df = pd.concat(
            [users_df] + [pd.DataFrame({col: vals}) for col, vals in new_cols.items()],
            axis=1,
        )
        data["users"] = users_df

    return add_moving_average


def add_basic_interaction_statistics(
    question_types: list[QuestionType], question_purposes: list[QuestionPurpose]
):
    """
    Add basic interaction statistics to the users DataFrame.
    """

    def add_basic_interaction_statistics(data: Dict[str, pd.DataFrame]):
        users_df = data["users"]
        interactions_df = data["interactions"]

        new_cols = {}

        for qtype in question_types:
            col_name = f"num_{qtype.name}_questions"
            counts = interactions_df[interactions_df["question_type_by_ai"] == qtype][
                "user_id"
            ].value_counts()
            new_cols[col_name] = users_df["user_id"].map(counts).fillna(0).astype(int)

        for qpurpose in question_purposes:
            col_name = f"num_{qpurpose.name}_questions"
            counts = interactions_df[
                interactions_df["question_purpose_by_question_type"] == qpurpose
            ]["user_id"].value_counts()
            new_cols[col_name] = users_df["user_id"].map(counts).fillna(0).astype(int)

        # Assign all new columns at once to avoid fragmentation
        if new_cols:
            users_df = pd.concat(
                [users_df, pd.DataFrame(new_cols, index=users_df.index)], axis=1
            )

        data["users"] = users_df

    return add_basic_interaction_statistics


def add_aggregate_construct_series(data: Dict[str, pd.DataFrame]) -> None:
    """
    For each user, aggregate all individual construct series into a single series
    that contains all construct-related practice events (datetime, result).
    Adds a new column `all_constructs_series` to `users` where each cell is a
    DataFrame with columns `datetime` and `result`.
    """
    users_df = data["users"]

    # find construct series columns (created by add_construct_result_series)
    construct_cols = [c for c in users_df.columns if c.endswith("_construct_series")]

    def aggregate_series_for_row(row):
        dfs = []
        for col in construct_cols:
            cell = row.get(col) if isinstance(row, dict) else row[col]
            # cell is expected to be a DataFrame or something similar
            if isinstance(cell, pd.DataFrame) and not cell.empty:
                dfs.append(cell)
        if not dfs:
            return pd.DataFrame(columns=["datetime", "result"])
        combined = pd.concat(dfs).sort_values("datetime").reset_index(drop=True)
        return combined

    # Use apply on rows to keep each user's aggregated series
    users_df["all_constructs_series"] = users_df.apply(aggregate_series_for_row, axis=1)
    data["users"] = users_df


def add_aggregate_learning_goal_series(learning_goals: list[LearningGoal]):
    """
    Returns a function that when given `data` will aggregate all per-learning-goal
    series columns (created by `add_learning_goals_result_series`) into a single
    `all_learning_goals_series` column per user.
    """

    def add_agg(data: Dict[str, pd.DataFrame]) -> None:
        users_df = data["users"]

        # find learning goal series columns (e.g., <GOAL>_series)
        goal_cols = [f"{goal.name}_series" for goal in learning_goals]

        def aggregate_series_for_row(row):
            dfs = []
            for col in goal_cols:
                # be tolerant: if column doesn't exist, skip
                if col not in users_df.columns:
                    continue
                cell = row.get(col) if isinstance(row, dict) else row[col]
                if isinstance(cell, pd.DataFrame) and not cell.empty:
                    dfs.append(cell)
            if not dfs:
                return pd.DataFrame(columns=["datetime", "result"])
            combined = pd.concat(dfs).sort_values("datetime").reset_index(drop=True)
            return combined

        users_df["all_learning_goals_series"] = users_df.apply(
            aggregate_series_for_row, axis=1
        )
        data["users"] = users_df

    return add_agg


def add_basic_statistics_for_series(series_column: str):
    """
    Factory that returns a function which computes basic statistics (average, slope,
    counts) on a per-user series column `series_column` and stores them with names
    prefixed by `prefix` (e.g., `prefix_average_success`).
    """

    def add_stats(data: Dict[str, pd.DataFrame]) -> None:
        users_df = data["users"]

        def compute_average(user_result_df):
            if (
                user_result_df is None
                or not isinstance(user_result_df, pd.DataFrame)
                or user_result_df.empty
            ):
                return None
            num_true = (user_result_df["result"] == True).sum()
            num_false = (user_result_df["result"] == False).sum()
            total = num_true + num_false
            if total == 0:
                return None
            return num_true / total

        def compute_slope(user_result_df):
            if (
                user_result_df is None
                or not isinstance(user_result_df, pd.DataFrame)
                or len(user_result_df) < 2
            ):
                return None
            x = user_result_df["datetime"].map(lambda dt: dt.timestamp()).values
            y = user_result_df["result"].astype(float).values
            slope, intercept, r, p, se = linregress(x, y)
            return slope

        def compute_num_practices(user_result_df):
            if (
                user_result_df is None
                or not isinstance(user_result_df, pd.DataFrame)
                or user_result_df.empty
            ):
                return 0
            return len(user_result_df)

        def compute_num_successes(user_result_df):
            if (
                user_result_df is None
                or not isinstance(user_result_df, pd.DataFrame)
                or user_result_df.empty
            ):
                return 0
            return (user_result_df["result"] == True).sum()

        def compute_num_failures(user_result_df):
            if (
                user_result_df is None
                or not isinstance(user_result_df, pd.DataFrame)
                or user_result_df.empty
            ):
                return 0
            return (user_result_df["result"] == False).sum()

        users_df[f"{series_column}_average_success"] = users_df[series_column].apply(
            compute_average
        )
        users_df[f"{series_column}_slope"] = users_df[series_column].apply(compute_slope)
        users_df[f"{series_column}_num_practices"] = users_df[series_column].apply(
            compute_num_practices
        )
        users_df[f"{series_column}_num_successes"] = users_df[series_column].apply(
            compute_num_successes
        )
        users_df[f"{series_column}_num_failures"] = users_df[series_column].apply(
            compute_num_failures
        )

        data["users"] = users_df.copy()

    return add_stats
