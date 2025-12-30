from collections.abc import Generator
from typing import Optional, Union

from dify_plugin.entities.model.llm import LLMMode, LLMResult
from dify_plugin.entities.model.message import PromptMessage, PromptMessageTool
from dify_plugin import OAICompatLargeLanguageModel


class HyperCLOVAXLargeLanguageModel(OAICompatLargeLanguageModel):
    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        # Initialize custom parameters and feature flags
        self._add_custom_parameters(credentials, model)

        # HyperCLOVA X supports repetition_penalty and top_k via the OpenAI-compatible endpoint.
        # These are passed directly within model_parameters.

        return super()._invoke(
            model, credentials, prompt_messages, model_parameters, tools, stop, stream
        )

    def validate_credentials(self, model: str, credentials: dict) -> None:
        self._add_custom_parameters(credentials, model)
        super().validate_credentials(model, credentials)

    @staticmethod
    def _add_custom_parameters(credentials: dict, model: str) -> None:
        """
        Configures the credentials to point to the NAVER Cloud OpenAI-compatible endpoint
        and sets feature support based on the specific HyperCLOVA X model.
        """
        # Base URL for CLOVA Studio OpenAI Compatibility
        credentials["endpoint_url"] = "https://clovastudio.stream.ntruss.com/v1/openai"

        credentials["mode"] = LLMMode.CHAT.value

        # Tool Call (Function Calling) Support
        # Supported by HCX-007, HCX-005, and HCX-DASH-002
        if model in ["HCX-007", "HCX-005", "HCX-DASH-002"]:
            credentials["function_calling_type"] = "tool_call"
            credentials["stream_function_calling"] = "support"
        else:
            # Legacy models (HCX-003, HCX-DASH-001) do not support function calling
            credentials["function_calling_type"] = None
            credentials["stream_function_calling"] = "no_support"

        # Vision Support (Multimodal)
        # Only HCX-005 supports image inputs
        if model == "HCX-005":
            credentials["vision_support"] = "support"
        else:
            credentials["vision_support"] = "no_support"

    def _update_model_parameters(self, model: str, model_parameters: dict):
        """
        Custom parameter mapping for HCX specific fields.
        """
        # HCX-007 uses reasoning_effort in the OpenAI compatible layer
        # to trigger the 'Thinking' process.
        if model == "HCX-007" and "reasoning_effort" not in model_parameters:
            # Default to medium reasoning if not specified for the reasoning model
            model_parameters["reasoning_effort"] = "medium"

        # Ensure max_tokens is mapped to max_completion_tokens for compatibility if needed
        if "max_tokens" in model_parameters and model == "HCX-007":
            model_parameters["max_completion_tokens"] = model_parameters.pop(
                "max_tokens"
            )

        return model_parameters
