from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from bardapi import Bard


class BardCustom(LLM):

    chatbot = Bard()

    @property
    def _llm_type(self) -> str:
        return "bard_test"

    def _call(self,
                prompt: str,
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return self.chatbot.get_answer(prompt)['content']

    # @property
    # def _identifying_params(self) -> Mapping[str, Any]:
    #     """Get the identifying parameters from initialization."""
    #     return {"n": self.n}


if __name__ == "__main__":
    print(Bard().get_answer("Hi how are you")['content'])

