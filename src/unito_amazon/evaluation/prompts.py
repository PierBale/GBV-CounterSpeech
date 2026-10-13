"""JudgeLM pairwise prompt adapted to the model's native evaluation format."""


PAIRWISE_EVALUATION_PROMPT = """You are an impartial and precise evaluator comparing two counter-speech responses to a hateful message.

Evaluate both responses according to these criteria, in this exact order of importance:
1. Factuality: are the claims accurate and verifiable? Factual correctness is the highest priority.
2. Number of facts: does the response provide multiple relevant facts? Prefer more well-chosen facts.
3. Relevance: does it directly address the hateful message?
4. Specificity: is it specific rather than generic?

Assign each response an overall integer score from 1 to 10 based on these criteria. A higher score means better overall counter-speech. Use the full scale when warranted; do not give equal scores merely because both responses are acceptable. Give equal scores only when neither response is better under the ordered criteria. Judge the content, not the position or wording of the response labels."""

JSON_OUTPUT_INSTRUCTION = """Return exactly one valid JSON object and no Markdown or surrounding text. It must contain exactly these fields:
- "score_response_1": an integer from 1 to 10
- "score_response_2": an integer from 1 to 10
- "winner": a string whose value is response_1, response_2, or tie
- "reasoning": a concise explanation of the comparison based on the criteria

The winner must agree with the scores: the response with the higher score wins; use tie only when the two scores are equal. Do not add other fields."""


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
{PAIRWISE_EVALUATION_PROMPT}

{JSON_OUTPUT_INSTRUCTION}"""
