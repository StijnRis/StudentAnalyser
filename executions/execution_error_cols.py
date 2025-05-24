from enum import Enum


class ExecutionErrorCols(Enum):
    ID = "id"
    EXECUTION_ID = "execution_id"
    ERROR_NAME = "error_name"
    ERROR_VALUE = "error_value"
    TRACEBACK = "traceback"
    LEARNING_GOALS_IN_ERROR_BY_ERROR_PATTERN_DETECTION = (
        "learning_goals_in_error_by_error_pattern_detection"
    )
    LEARNING_GOALS_IN_ERROR_BY_AI_DETECTION = "learning_goals_in_error_by_ai_detection"
    LEARNING_GOALS_IN_ERROR_BY_AI_DETECTION_PROMPT = (
        "learning_goals_in_error_by_ai_detection_prompt"
    )
    LEARNING_GOALS_IN_ERROR_BY_AI_DETECTION_RESPONSE = (
        "learning_goals_in_error_by_ai_detection_response"
    )
