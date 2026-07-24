from __future__ import annotations

TASK_B_LABELS = [
    "1. threats, plans to harm and incitement",
    "2. derogation",
    "3. animosity",
    "4. prejudiced discussions",
]

TASK_C_LABELS = [
    "1.1 threats of harm",
    "1.2 incitement and encouragement of harm",
    "2.1 descriptive attacks",
    "2.2 aggressive and emotive attacks",
    "2.3 dehumanising attacks & overt sexual objectification",
    "3.1 casual use of gendered slurs, profanities, and insults",
    "3.2 immutable gender differences and gender stereotypes",
    "3.3 backhanded gendered compliments",
    "3.4 condescending explanations or unwelcome advice",
    "4.1 supporting mistreatment of individual women",
    "4.2 supporting systemic discrimination against women as a group",
]

TASK_C_TO_B = {
    "1.1 threats of harm": "1. threats, plans to harm and incitement",
    "1.2 incitement and encouragement of harm": "1. threats, plans to harm and incitement",
    "2.1 descriptive attacks": "2. derogation",
    "2.2 aggressive and emotive attacks": "2. derogation",
    "2.3 dehumanising attacks & overt sexual objectification": "2. derogation",
    "3.1 casual use of gendered slurs, profanities, and insults": "3. animosity",
    "3.2 immutable gender differences and gender stereotypes": "3. animosity",
    "3.3 backhanded gendered compliments": "3. animosity",
    "3.4 condescending explanations or unwelcome advice": "3. animosity",
    "4.1 supporting mistreatment of individual women": "4. prejudiced discussions",
    "4.2 supporting systemic discrimination against women as a group": "4. prejudiced discussions",
}

def labels_for_task(task: str):
    task = task.lower()
    if task == "b":
        return TASK_B_LABELS
    if task == "c":
        return TASK_C_LABELS
    raise ValueError("task must be 'b' or 'c'")
