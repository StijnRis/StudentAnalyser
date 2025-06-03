from typing import Dict

import pandas as pd


def load_stanislas_grades(
    log_path: str, max_points: int, group: str, filter_username: str | None
):
    def load_stanislas_grades(data: Dict[str, pd.DataFrame]) -> None:
        df = pd.read_csv(log_path)
        df["total_points"] = df.iloc[:, 1:].sum(axis=1)
        df["grade"] = df["total_points"] / max_points * 9 + 1
        users = data.get(
            "users", pd.DataFrame(columns=["user_id", "group", "username"])
        )
        if "grade" not in users.columns:
            users["grade"] = pd.NA
        user_map = {
            (r["group"], r["username"]): r["user_id"] for _, r in users.iterrows()
        }
        next_id = users["user_id"].max() + 1 if not users.empty else 0
        for username, grade in zip(df["Leerlingnummer"].astype(str), df["grade"]):
            if filter_username and username != filter_username:
                continue
            key = (group, username)
            if key in user_map:
                user_row = users[
                    (users["group"] == group) & (users["username"] == username)
                ]
                if not user_row["grade"].isna().all():
                    raise Exception(f"User {key} already has a grade!")
                users.loc[user_row.index, "grade"] = grade
            else:
                users = pd.concat(
                    [
                        users,
                        pd.DataFrame(
                            {
                                "user_id": [next_id],
                                "group": [group],
                                "username": [username],
                                "grade": [grade],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
                user_map[key] = next_id
                next_id += 1
        data["users"] = users

    return load_stanislas_grades
