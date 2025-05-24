from enum import Enum


class ExecutionSuccessCols(Enum):
    ID = "id"
    EXECUTION_ID = "execution_id"
    LINE_NUMBERS_OF_NEW_CODE = "line_numbers_of_new_code"
    ADDED_CONSTRUCTS = "added_constructs"
    LEARNING_GOALS_OF_ADDED_CODE = "learning_goals_of_added_code"