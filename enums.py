import ast
import re
from typing import Callable, List


class LearningGoal:
    def __init__(
        self,
        name: str,
        description: str,
        is_applied: Callable[[ast.AST], bool],
        found_in_error: Callable[[str, str, str, str], bool],
    ):
        self.name = name
        self.description = description
        self.is_applied_lambda = is_applied
        self.found_in_error_lambda = found_in_error

    def is_applied(self, node: ast.AST) -> bool:
        return self.is_applied_lambda(node)

    def found_in_error(
        self, error_name: str, traceback: str, code: str, code_line: str
    ) -> bool:
        return self.found_in_error_lambda(error_name, traceback, code, code_line)

    def __str__(self):
        return f"{self.name}"


class QuestionPurpose:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def __str__(self):
        return f"{self.name}"


class QuestionType:
    def __init__(self, name: str, description: str, question_purpose: QuestionPurpose):
        self.name = name
        self.description = description
        self.question_purpose = question_purpose

    def __str__(self):
        return f"{self.name}"


executive_purpose = QuestionPurpose(
    "Executive",
    "The students asks the chatbot to complete a task for them.",
)

instrumental_purpose = QuestionPurpose(
    "Instrumental",
    "The students ask for minimal help so they can complete a task themselves.",
)

not_detected_purpose = QuestionPurpose(
    "Not detected",
    "Unable to detect the question's purpose.",
)


def get_question_purposes() -> List[QuestionPurpose]:
    return [
        executive_purpose,
        instrumental_purpose,
        not_detected_purpose,
    ]


def get_question_types() -> List[QuestionType]:
    """
    Returns a list of question types.
    """
    return [
        QuestionType(
            "Chatting with the chatbot",
            "The user is engaging in a conversation with the chatbot, not asking a specific question.",
            not_detected_purpose,
        ),
        QuestionType(
            "Code comprehension",
            "The user is asking for an explanation of a piece of code, how it works, what it does, or why it is producing unexpected results.",
            instrumental_purpose,
        ),
        QuestionType(
            "Concept comprehension",
            "The user is requesting an explanation of a concept, method, technique, idea, or underlying principle.",
            instrumental_purpose,
        ),
        QuestionType(
            "Error comprehension",
            "The user is asking for help in understanding a specific error message (e.g., pasting an error message and asking what it means).",
            instrumental_purpose,
        ),
        QuestionType(
            "Problem comprehension",
            "The user is asking for clarification about a problem itself, such as what it means or how to interpret it.",
            instrumental_purpose,
        ),
        QuestionType(
            "Copied problem",
            "The user has copied and pasted the problem. These often appear as a set of instructions.",
            executive_purpose,
        ),
        QuestionType(
            "Fix code",
            "The user is asking the chatbot to fix broken code.",
            executive_purpose,
        ),
        QuestionType(
            "Task delegation",
            "The user is asking the chatbot to perform a task, such as generating code or calculating something.",
            executive_purpose,
        ),
        QuestionType(
            "Pasted code without explanation",
            "The user has pasted a piece of code without providing an explicit question or any context.",
            executive_purpose,
        ),
        QuestionType(
            "Not detected",
            "Unable to determine the type of question due to ambiguity, lack of information or not fitting any predefined category.",
            not_detected_purpose,
        ),
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
            lambda error_name, traceback, code, error_line: (
                "syntaxerror" in error_name.lower()
                and bool(re.search(r"\bprint\b", error_line.lower()))
            ),
        ),
        LearningGoal(
            "Input statement",
            "Using the input function.",
            lambda node: (
                isinstance(node, ast.Call)
                and isinstance(getattr(node, "func", None), ast.Name)
                and getattr(node.func, "id", None) == "input"
            ),
            lambda error_name, traceback, code, error_line: (
                "syntaxerror" in error_name.lower()
                and bool(re.search(r"\binput\b", error_line.lower()))
            ),
        ),
        LearningGoal(
            "Variable assignment",
            "Assigning values to variables.",
            lambda node: isinstance(node, (ast.Assign, ast.AugAssign)),
            lambda error_name, traceback, code, error_line: (
                "syntaxerror" in error_name.lower()
                and (
                    "cannot assign" in traceback.lower()
                    or "assignment" in traceback.lower()
                    or "can't assign" in traceback.lower()
                )
            ),
        ),
        LearningGoal(
            "Variable usage",
            "Using variables in expressions or statements.",
            lambda node: (
                isinstance(node, ast.Name)
                and isinstance(getattr(node, "ctx", None), ast.Load)
                and not isinstance(getattr(node, "ctx", None), (ast.Store, ast.Del))
            ),
            lambda error_name, traceback, code, error_line: (
                "nameerror" in error_name.lower()
                and "is not defined" in traceback.lower()
            ),
        ),
        LearningGoal(
            "Conditionals",
            "Using if/else statements.",
            lambda node: isinstance(node, ast.If),
            lambda error_name, traceback, code, error_line: (
                "syntaxerror" in error_name.lower()
                and bool(re.search(r"\b(if|else|elif)\b", error_line.lower()))
            ),
        ),
        LearningGoal(
            "For loop",
            "Error with a for loop.",
            lambda node: isinstance(node, ast.For),
            lambda error_name, traceback, code, error_line: (
                "syntaxerror" in error_name.lower()
                and bool(re.search(r"\bfor\b", error_line.lower()))
            ),
        ),
        LearningGoal(
            "While loop",
            "Error with a while loop.",
            lambda node: isinstance(node, ast.While),
            lambda error_name, traceback, code, error_line: (
                "syntaxerror" in error_name.lower()
                and bool(re.search(r"\bwhile\b", error_line.lower()))
            ),
        ),
        LearningGoal(
            "Break statement",
            "Error with a break statement.",
            lambda node: isinstance(node, ast.Break),
            lambda error_name, traceback, code, error_line: (
                "syntaxerror" in error_name.lower()
                and bool(re.search(r"\bbreak\b", error_line.lower()))
            ),
        ),
        LearningGoal(
            "Function call",
            "Error with a function call.",
            lambda node: isinstance(node, ast.Call),
            lambda error_name, traceback, code, error_line: (
                "attributeerror" in error_name.lower()
            ),
        ),
        LearningGoal(
            "Function definition",
            "Error with a function definition.",
            lambda node: isinstance(node, ast.FunctionDef),
            lambda error_name, traceback, code, error_line: (
                "syntaxerror" in error_name.lower()
                and bool(re.search(r"\bdef\b", error_line.lower()))
            ),
        ),
        LearningGoal(
            "Import statement",
            "Error with an import statement.",
            lambda node: isinstance(node, (ast.Import, ast.ImportFrom)),
            lambda error_name, traceback, code, error_line: (
                "syntaxerror" in error_name.lower()
                and bool(re.search(r"\bimport\b", error_line.lower()))
            ),
        ),
        LearningGoal(
            "List access",
            "Error with accessing a list.",
            lambda node: isinstance(node, ast.Subscript),
            lambda error_name, traceback, code, error_line: (
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
            lambda error_name, traceback, code, error_line: (
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
            lambda error_name, traceback, code, error_line: (
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
            lambda error_name, traceback, code, code_line: (
                "typeerror" in error_name.lower()
            ),
        ),
        LearningGoal(
            "Typo",
            "This learning goal is applied when a typo is detected in the code.",
            lambda node: False,
            lambda error_name, traceback, code, code_line: (
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
            lambda error_name, traceback, code, code_line: False,
        ),
    ]
