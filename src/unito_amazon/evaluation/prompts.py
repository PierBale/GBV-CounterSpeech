"""JudgeLM pairwise prompt adapted to the model's native evaluation format."""

PAIRWISE_EVALUATION_PROMPT = """You are an impartial evaluator. You will be given a piece of hate speech and two counter-speech responses. Your task is to judge which counter-speech is better according to the following criteria, in order of importance:
1) Factuality - Does the response contain accurate and verifiable information? A factually correct response is always preferred.
2) Number of facts - Does the response provide multiple relevant facts, not just one? More well-chosen facts are better.
3) Relevance - Does the response directly address the hate speech?
4) Specificity - Is the response specific and non-generic?"""

SCORE_OUTPUT_INSTRUCTION = """Each assistant receives an overall score on a scale of 1 to 10 based on the ordered criteria above, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""


class CounterSpeechPairwisePrompt:
    """Prompt for pairwise counter-speech evaluation in JudgeLM's learned format."""

    system_prompt = ""

    def build(
        self,
        hateful_message: str,
        response_1: str,
        response_2: str,
    ) -> str:
        return f"""[Question]
Hateful message: {hateful_message}

[The Start of Assistant 1's Answer]
{response_1}

[The End of Assistant 1's Answer]

[The Start of Assistant 2's Answer]
{response_2}

[The End of Assistant 2's Answer]

[System]
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.

{PAIRWISE_EVALUATION_PROMPT}

{SCORE_OUTPUT_INSTRUCTION}"""
