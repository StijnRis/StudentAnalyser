import ast
from typing import Callable, List


class LearningGoal:
    def __init__(
        self,
        name: str,
        description: str,
        is_applied: Callable[[ast.AST], bool],
        found_in_error: Callable[[str, str, str], bool],
    ):
        self.name = name
        self.description = description
        self.is_applied = is_applied
        self.found_in_error_lambda = found_in_error

    def found_in_error(self, error_name, traceback, code):
        return self.found_in_error_lambda(error_name, traceback, code)

    def __str__(self):
        return f"{self.name}"


class QuestionPurpose:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def __str__(self):
        return f"{self.name}"


class QuestionType:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def __str__(self):
        return f"{self.name}"


def get_question_types() -> List[QuestionType]:
    """
    Returns a list of question types.
    """
    return [
        QuestionType(
            "Answer to question of chatbot",
            "The user is responding to a question asked by the chatbot rather than asking a new one.",
        ),
        QuestionType(
            "Code comprehension",
            "The user is asking for an explanation of a piece of code, how it works, or what it does.",
        ),
        QuestionType(
            "Concept comprehension",
            "The user is asking for an explanation of a theoretical or abstract concept, often related to programming, science, or another domain.",
        ),
        QuestionType(
            "Error comprehension",
            "The user is asking for help in understanding an error message, its cause, and how to fix it.",
        ),
        QuestionType(
            "Question comprehension",
            "The user is asking for clarification about a question itself, such as what it means or how to interpret it.",
        ),
        QuestionType(
            "Copied question",
            "The user has copied and pasted a question from another source without modification.",
        ),
        QuestionType(
            "Fix code",
            "The user is asking for help in correcting a bug, syntax issue, or logical error in a piece of code.",
        ),
        QuestionType(
            "Task delegation",
            "The user is asking the chatbot to perform a specific task, such as generating code, writing a document, or performing an analysis.",
        ),
        QuestionType(
            "Pasted code without context",
            "The user has pasted a piece of code without providing an explicit question or any context.",
        ),
        QuestionType(
            "Other",
            "The user's question does not fit into any of the predefined categories.",
        ),
        QuestionType(
            "Not detected",
            "Unable to determine the type of question due to ambiguity or lack of information.",
        ),
    ]


def get_question_purposes() -> List[QuestionPurpose]:
    return [
        QuestionPurpose(
            "Executive",
            "The question is about planning, organizing, or managing tasks or activities.",
        ),
        QuestionPurpose(
            "Instrumental",
            "The question is about carrying out specific actions, solving problems, or using tools to achieve a goal.",
        ),
        QuestionPurpose("Not detected", "Unable to detect the question's purpose."),
    ]


def get_learning_goals() -> List[LearningGoal]:
    return [
        LearningGoal(
            "Print statement",
            "Using the print statement.",
            lambda node: (
                isinstance(node, ast.Call)
                and isinstance(getattr(node, "func", None), ast.Name)
                and getattr(node.func, "id", None) == "print"
            ),
            lambda error_name, traceback, code: (
                "syntaxerror" in error_name.lower() and "print" in traceback.lower()
            ),
        ),
        LearningGoal(
            "Variable assignment",
            "Assigning values to variables.",
            lambda node: isinstance(node, (ast.Assign, ast.AugAssign)),
            lambda error_name, traceback, code: (
                "nameerror" in error_name.lower() and "not defined" in traceback.lower()
            ),
        ),
        LearningGoal(
            "Conditionals",
            "Using if/else statements.",
            lambda node: isinstance(node, ast.If),
            lambda error_name, traceback, code: False,
        ),
        LearningGoal(
            "For loop",
            "Error with a for loop.",
            lambda node: isinstance(node, ast.For),
            lambda error_name, traceback, code: (
                "syntaxerror" in error_name.lower() and "for" in traceback.lower()
            ),
        ),
        LearningGoal(
            "While loop",
            "Error with a while loop.",
            lambda node: isinstance(node, ast.While),
            lambda error_name, traceback, code: (
                "syntaxerror" in error_name.lower() and "while" in traceback.lower()
            ),
        ),
        LearningGoal(
            "Break statement",
            "Error with a break statement.",
            lambda node: isinstance(node, ast.Break),
            lambda error_name, traceback, code: (
                "syntaxerror" in error_name.lower() and "break" in traceback.lower()
            ),
        ),
        LearningGoal(
            "Function call",
            "Error with a function call.",
            lambda node: isinstance(node, ast.Call),
            lambda error_name, traceback, code: (
                "attributeerror" in error_name.lower()
            ),
        ),
        LearningGoal(
            "Function definition",
            "Error with a function definition.",
            lambda node: isinstance(node, ast.FunctionDef),
            lambda error_name, traceback, code: (
                "syntaxerror" in error_name.lower() and "def" in traceback.lower()
            ),
        ),
        LearningGoal(
            "Import statement",
            "Error with an import statement.",
            lambda node: isinstance(node, (ast.Import, ast.ImportFrom)),
            lambda error_name, traceback, code: (
                "syntaxerror" in error_name.lower() and "import" in traceback.lower()
            ),
        ),
        LearningGoal(
            "List access",
            "Error with accessing a list.",
            lambda node: isinstance(node, ast.Subscript),
            lambda error_name, traceback, code: (
                "indexerror" in error_name.lower()
                or (
                    "typeerror" in error_name.lower()
                    and "object is not subscriptable" in error_name.lower()
                )
                or "keyerror" in error_name.lower()
            ),
        ),
        LearningGoal(
            "List assignment",
            "Error with setting a value in a list.",
            lambda node: isinstance(node, ast.Assign)
            and isinstance(getattr(node, "targets", [None])[0], ast.Subscript),
            lambda error_name, traceback, code: (
                "indexerror" in error_name.lower()
                or (
                    "typeerror" in error_name.lower()
                    and "object is not subscriptable" in error_name.lower()
                )
                or "keyerror" in error_name.lower()
            ),
        ),
        LearningGoal(
            "List declaration",
            "Error with defining a list.",
            lambda node: isinstance(node, ast.List),
            lambda error_name, traceback, code: (
                "indexerror" in error_name.lower()
                or (
                    "typeerror" in error_name.lower()
                    and "object is not subscriptable" in error_name.lower()
                )
                or "keyerror" in error_name.lower()
            ),
        ),
        LearningGoal(
            "Type casting",
            "Operation involving data types.",
            lambda node: (
                isinstance(node, ast.Call)
                and isinstance(getattr(node, "func", None), ast.Name)
                and getattr(node.func, "id", None)
                in {"int", "float", "str", "bool", "list", "dict", "set", "tuple"}
            ),
            lambda error_name, traceback, code: ("typeerror" in error_name.lower()),
        ),
        LearningGoal(
            "Typo",
            "This learning goal is applied when a typo is detected in the code.",
            lambda node: False,
            lambda error_name, traceback, code: (
                ("syntaxerror" in error_name.lower() and "eol" in error_name.lower())
                or (
                    "syntaxerror" in error_name.lower()
                    and (
                        "unexpected eof" in error_name.lower()
                        or "unterminated string literal" in error_name.lower()
                    )
                )
            ),
        ),
        LearningGoal(
            "Not detected",
            "Unable to detect the learning goal.",
            lambda node: False,
            lambda error_name, traceback, code: False,
        ),
    ]
