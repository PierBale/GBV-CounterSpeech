PAPER_RAG_VS_NO_RAG_PROMPT = """Provide a brief counter-narrative in response to the following hate speech: {hs}. You are an impartial evaluator. You will be given a piece of hate speech and two counter-speech responses. Your task is to judge which counter-speech is better according to the following criteria, in order of importance:
1) Factuality – Does the response contain accurate and verifiable information? A factually correct response is always preferred. 2) Number of facts – Does the response provide multiple relevant facts, not just one? More well-chosen facts are better. 3) Relevance – Does the response directly address the hate speech? 4) Specificity – Is the response specific and non-generic?"""


class CounterSpeechPairwisePrompt:
    """RAG-vs-No-RAG JudgeLM prompt from Damo et al. (2025)."""

    system_prompt = ""

    def build(
        self,
        hateful_message: str,
        response_1: str,
        response_2: str,
    ) -> str:
        instruction = PAPER_RAG_VS_NO_RAG_PROMPT.format(hs=hateful_message)
        return f"""[Question]
{instruction}
[End of Question]

[Assistant 1]
{response_1}
[End of Assistant 1]

[Assistant 2]
{response_2}
[End of Assistant 2]"""
