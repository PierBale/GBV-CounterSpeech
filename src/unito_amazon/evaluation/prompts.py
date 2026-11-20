"""JudgeLM pairwise prompt adapted to the model's native evaluation format.

Two sets of criteria, both from Damo et al. (2025, arXiv:2510.12316, Appendix B):
- "facts" (default): factuality, number of facts, relevance, specificity, in order of
  importance (their RAG vs No-RAG judge prompt; used for data/counterspeech_v2);
- "social_media": factual, easy to read, empathetic, colloquial and about 2 sentences
  long (their judge prompt for the comparison with other RAG-based CS systems).
"""

PAIRWISE_EVALUATION_PROMPT = """You are an impartial evaluator. You will be given a piece of hate speech and two counter-speech responses. Your task is to judge which counter-speech is better according to the following criteria, in order of importance:
1) Factuality - Does the response contain accurate and verifiable information? A factually correct response is always preferred.
2) Number of facts - Does the response provide multiple relevant facts, not just one? More well-chosen facts are better.
3) Relevance - Does the response directly address the hate speech?
4) Specificity - Is the response specific and non-generic?"""

SCORE_OUTPUT_INSTRUCTION = """Each assistant receives an overall score on a scale of 1 to 10 based on the ordered criteria above, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""

SOCIAL_MEDIA_EVALUATION_PROMPT = """You are an impartial evaluator. You will be given a piece of hate speech and two counter-speech responses. Your task is to judge which response is better according to the following criteria:
- The response must be factual, easy to read, empathetic, and suitable as a short social media comment.
- Friendly and colloquial language is preferred. Responses that sound like essays, lectures, or academic articles should be penalized.
- Conciseness: The response should be around 2 sentences long."""

CRITERIA = {
    "facts": (PAIRWISE_EVALUATION_PROMPT, SCORE_OUTPUT_INSTRUCTION),
    "social_media": (SOCIAL_MEDIA_EVALUATION_PROMPT,
                     SCORE_OUTPUT_INSTRUCTION.replace("based on the ordered criteria above",
                                                      "based on the criteria above")),
}


class CounterSpeechPairwisePrompt:
    """Prompt for pairwise counter-speech evaluation in JudgeLM's learned format."""

    system_prompt = ""

    def __init__(self, criteria: str = "facts"):
        if criteria not in CRITERIA:
            raise ValueError(f"Unknown judge criteria {criteria!r}; expected one of {sorted(CRITERIA)}.")
        self.criteria = criteria
        self.evaluation_prompt, self.score_instruction = CRITERIA[criteria]

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

{self.evaluation_prompt}

{self.score_instruction}"""
