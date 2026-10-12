PAIRWISE_EVALUATION_PROMPT = """You are an impartial evaluator. You will be given a piece of hate speech and two counter-speech responses. Your task is to judge which counter-speech is better according to the following criteria, in order of importance:
1) Factuality – Does the response contain accurate and verifiable information? A factually correct response is always preferred.
2) Number of facts – Does the response provide multiple relevant facts, not just one? More well-chosen facts are better.
3) Relevance – Does the response directly address the hate speech?
4) Specificity – Is the response specific and non-generic?"""

JSON_OUTPUT_INSTRUCTION = """Return only one valid JSON object, with no Markdown or additional text, using exactly this schema:
{"winner": "<winner>"}

Replace "<winner>" with exactly one of "response_1", "response_2", or "tie". Use "tie" only when neither response is better according to the ordered criteria."""


class CounterSpeechPairwisePrompt:
    """Prompt for pairwise counter-speech evaluation."""

    system_prompt = ""

    def build(
        self,
        hateful_message: str,
        response_1: str,
        response_2: str,
    ) -> str:
        return f"""{PAIRWISE_EVALUATION_PROMPT}

[Hate Speech]
{hateful_message}
[End of Hate Speech]

[Response 1]
{response_1}
[End of Response 1]

[Response 2]
{response_2}
[End of Response 2]

{JSON_OUTPUT_INSTRUCTION}"""
