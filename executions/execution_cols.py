from enum import Enum


class ExecutionsCols(Enum):
    ID = "id"
    USERNAME = "username"
    DATETIME = "datetime"
    FILE = "file"
    FILE_VERSION_ID = "file_version_id"
    SUCCESS = "success"
    PREVIOUS_SUCCESS_ID = "previous_success_id"
    NEXT_SUCCESS_ID = "next_success_id"
    PREVIOUS_ERROR_ID = "previous_error_id"
    NEXT_ERROR_ID = "next_error_id"
    PREV_EXECUTED_FILE_VERSION_ID = "prev_executed_file_version_id"
    EXECUTION_ID = "execution_id"
