from __future__ import annotations

import os
import re
import warnings
from typing import Optional, Tuple

try:
    from openai import OpenAI, AzureOpenAI  # type: ignore
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore
    AzureOpenAI = None  # type: ignore

try:  # pragma: no cover - compatibility for legacy SDKs
    import openai as _openai_legacy  # type: ignore
except ImportError:  # pragma: no cover
    _openai_legacy = None  # type: ignore


class LLMRechecker:
    _ANNOUNCED = False

    def __init__(
        self, model: str, temperature: float = 0.0, base_url: str | None = None
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = 512
        self._client: Optional["OpenAI"] = None
        self._legacy: Optional[object] = None
        self._status_reason: Optional[str] = None
        if OpenAI is None and _openai_legacy is None:
            self._status_reason = (
                "Python package 'openai' is not installed; install it in the active environment."
            )
            return
        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENAI_KEY")
            or os.getenv("AZURE_OPENAI_KEY")
            or os.getenv("AZURE_OPENAI_API_KEY")
        )
        candidate_base = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_ENDPOINT")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        if candidate_base and "chatgpt.com" in candidate_base:
            warnings.warn(
                "OPENAI base URL points to chatgpt.com; disabling LLM rechecker to avoid unauthorized requests.",
                RuntimeWarning,
            )
            self._status_reason = "OPENAI_BASE_URL points to chatgpt.com, which is not an API endpoint."
            return
        api_type = (os.getenv("OPENAI_API_TYPE") or os.getenv("AZURE_OPENAI_API_TYPE") or "").lower()
        azure_endpoint = (
            candidate_base
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("OPENAI_ENDPOINT")
        )
        azure_deployment = (
            os.getenv("OPENAI_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        )
        api_version = (
            os.getenv("OPENAI_API_VERSION")
            or os.getenv("AZURE_OPENAI_API_VERSION")
            or os.getenv("OPENAI_API_VERSION")
        )

        if (api_type == "azure" or azure_deployment or os.getenv("AZURE_OPENAI_RESOURCE")) and AzureOpenAI is not None:
            client_kwargs = {
                "api_key": api_key,
            }
            if not client_kwargs["api_key"]:
                self._status_reason = "Azure OpenAI configuration requires AZURE_OPENAI_API_KEY (or OPENAI_API_KEY)."
                return
            if azure_endpoint:
                client_kwargs["azure_endpoint"] = azure_endpoint
            if api_version:
                client_kwargs["api_version"] = api_version
            try:
                self._client = AzureOpenAI(**client_kwargs)  # type: ignore[assignment]
                if azure_deployment:
                    self._model = azure_deployment
                self._status_reason = None
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"Failed to initialise Azure OpenAI client for LLM rechecker: {exc}",
                    RuntimeWarning,
                )
                self._client = None
                self._status_reason = f"Azure OpenAI client failed to initialise: {exc}"
        elif OpenAI is not None:
            client_kwargs = {}
            if api_key:
                client_kwargs["api_key"] = api_key
            if candidate_base and candidate_base.startswith("http"):
                client_kwargs["base_url"] = candidate_base
            try:
                self._client = OpenAI(**client_kwargs)
                self._status_reason = None
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"Failed to initialise OpenAI client for LLM rechecker: {exc}",
                    RuntimeWarning,
                )
                self._client = None
                self._status_reason = f"OpenAI client failed to initialise: {exc}"
        if self._client is None and _openai_legacy is not None:
            legacy_chat = getattr(_openai_legacy, "ChatCompletion", None)
            if legacy_chat is None or not hasattr(legacy_chat, "create"):
                if self._status_reason is None:
                    self._status_reason = (
                        "Legacy ChatCompletion interface unavailable in this OpenAI SDK."
                    )
            else:
                try:
                    if api_key:
                        _openai_legacy.api_key = api_key  # type: ignore[attr-defined]
                    if candidate_base:
                        setattr(_openai_legacy, "api_base", candidate_base)  # type: ignore[attr-defined]
                    if api_type == "azure" or azure_deployment:
                        setattr(_openai_legacy, "api_type", "azure")  # type: ignore[attr-defined]
                        if api_version:
                            setattr(_openai_legacy, "api_version", api_version)  # type: ignore[attr-defined]
                        if azure_deployment:
                            self._model = azure_deployment
                    self._legacy = _openai_legacy
                    self._status_reason = None
                except Exception as exc:  # pragma: no cover - configuration failure
                    warnings.warn(
                        f"Failed to configure legacy OpenAI client: {exc}",
                        RuntimeWarning,
                    )
                    self._legacy = None
                    if self._status_reason is None:
                        self._status_reason = f"Legacy OpenAI client failed to configure: {exc}"

        self._announce_if_ready()

    def _announce_if_ready(self) -> None:
        if LLMRechecker._ANNOUNCED:
            return
        provider = None
        if self._client is not None:
            if AzureOpenAI is not None and isinstance(self._client, AzureOpenAI):
                provider = "Azure OpenAI"
            else:
                provider = "OpenAI"
        elif self._legacy is not None:
            provider = "OpenAI (legacy)"
        if provider is None:
            return
        model_name = self._model or "(unspecified)"
        print(
            f"[LLM] OpenAI key detected; using {provider} model '{model_name}'.",
            flush=True,
        )
        LLMRechecker._ANNOUNCED = True

    @property
    def available(self) -> bool:
        if self._client is not None:
            return True
        if self._legacy is not None:
            legacy_chat = getattr(self._legacy, "ChatCompletion", None)
            if legacy_chat is None or not hasattr(legacy_chat, "create"):
                return False
            return True
        return False

    @property
    def status_reason(self) -> Optional[str]:
        return self._status_reason

    @staticmethod
    def _prepare_segment(segment: str | None) -> str:
        if not segment:
            return ""
        # Guard against braces interfering with str.format placeholders.
        return str(segment).replace("{", "{{").replace("}", "}}").strip()

    def validate(
        self, claim: str, quote: str, context: str
    ) -> Tuple[Optional[bool], Optional[str]]:
        if not self.available:
            return None, None
        prepared_claim = self._prepare_segment(claim)
        prepared_quote = self._prepare_segment(quote) or self._prepare_segment(context)
        prepared_context = self._prepare_segment(context)
        prompt = (
            "You are verifying whether a highlighted quote from a retrieved passage supports a factual claim.\n"
            "Treat phrases like 'virtual interactions', 'social VR', avatars sharing spaces, and similar descriptions as relevant to virtual reality social contexts when warranted.\n"
            "Claim: {claim}\n"
            "Highlighted quote:\n{quote}\n"
            "Full evidence passage:\n{context}\n"
            "State if the highlighted quote backs the claim.\n"
            "Respond using exactly two lines:\n"
            "Verdict: <YES|NO|UNKNOWN>\n"
            "Reason: <under 45 words; explain how the quote relates to the claim using details from the passage>"
        ).format(claim=prepared_claim, quote=prepared_quote, context=prepared_context)
        response = None
        if self._client is not None:
            params = {
                "model": self._model,
                "input": prompt,
                "max_output_tokens": self._max_output_tokens,
            }
            if self._temperature is not None:
                params["temperature"] = self._temperature
            try:
                response = self._client.responses.create(**params)
            except AttributeError:
                response, err_text = self._invoke_chat_api(prompt)
                if response is None:
                    return None, err_text
            except Exception as exc:  # pragma: no cover - network errors
                message = str(exc)
                if "Unsupported parameter" in message and "temperature" in message:
                    params.pop("temperature", None)
                    try:
                        response = self._client.responses.create(**params)
                    except Exception as inner_exc:  # pragma: no cover - network errors
                        err_text = self._normalise_error(inner_exc)
                        warnings.warn(
                            f"Failed to invoke responses API for LLM rechecker: {err_text}",
                            RuntimeWarning,
                        )
                        return None, f"LLM error: {err_text}"
                else:
                    err_text = self._normalise_error(exc)
                    warnings.warn(
                        f"Failed to invoke responses API for LLM rechecker: {err_text}",
                        RuntimeWarning,
                    )
                    return None, f"LLM error: {err_text}"
        elif self._legacy is not None:
            legacy_chat = getattr(self._legacy, "ChatCompletion", None)
            if legacy_chat is None or not hasattr(legacy_chat, "create"):
                message = (
                    "Legacy ChatCompletion API is not available in this OpenAI SDK version."
                )
                warnings.warn(message, RuntimeWarning)
                return None, f"LLM error: {message}"
            try:
                response = legacy_chat.create(  # type: ignore[attr-defined]
                    model=self._model,
                    messages=[
                        {"role": "system", "content": "You verify evidence for claims."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self._temperature,
                    max_tokens=self._max_output_tokens,
                )
            except Exception as exc:  # pragma: no cover - network errors
                err_text = self._normalise_error(exc)
                warnings.warn(
                    f"Failed to invoke legacy OpenAI chat API: {err_text}",
                    RuntimeWarning,
                )
                return None, f"LLM error: {err_text}"

        # Prefer the concatenated text provided by the SDK;
        # fall back to iterating the structured output if needed.
        candidates: list[str] = []
        try:
            combined_text = getattr(response, "output_text", "") or ""
            if combined_text:
                candidates.append(str(combined_text).strip())
        except Exception:
            combined_text = ""

        # Responses API structured output
        for item in getattr(response, "output", []) or []:
            if item.type != "message":
                continue
            parts: list[str] = []
            for content in getattr(item, "content", []) or []:
                text_value = getattr(content, "text", None)
                if hasattr(text_value, "value"):
                    text_str = str(getattr(text_value, "value"))
                else:
                    text_str = str(text_value) if text_value is not None else ""
                text_str = text_str.strip()
                if text_str:
                    parts.append(text_str)
            if parts:
                candidates.append(" ".join(parts).strip())

        # Chat completions style payloads
        for choice in getattr(response, "choices", []) or []:
            message = getattr(choice, "message", None)
            if not message:
                continue
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                candidates.append(content.strip())
            elif isinstance(content, list):
                combined = " ".join(
                    part.get("text", "").strip()
                    for part in content
                    if isinstance(part, dict)
                ).strip()
                if combined:
                    candidates.append(combined)

        for snippet in candidates:
            if not snippet:
                continue
            decision, rationale = self._interpret_reply(snippet)
            if decision is None and rationale is None:
                continue
            explanation = rationale or snippet
            return decision, explanation

        fallback = combined_text.strip() if combined_text else None
        if fallback:
            decision, rationale = self._interpret_reply(fallback)
            if decision is not None or rationale is not None:
                explanation = rationale or fallback
                return decision, explanation
        return None, fallback

    def _invoke_chat_api(self, prompt: str) -> tuple[Optional[object], str]:
        if self._client is None:
            return None, "LLM error: responses API unavailable and legacy client not initialised."
        chat_client = getattr(self._client, "chat", None)
        completions = getattr(chat_client, "completions", None) if chat_client else None
        if completions is None:
            return None, "LLM error: Client does not expose a compatible responses or chat API."
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": "You verify evidence for claims."},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": self._max_output_tokens,
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        try:
            return completions.create(**payload), ""
        except Exception as exc:  # pragma: no cover - network errors
            message = self._normalise_error(exc)
            if "Unsupported parameter" in message and "temperature" in message:
                payload.pop("temperature", None)
                try:
                    return completions.create(**payload), ""
                except Exception as inner_exc:  # pragma: no cover
                    inner_text = self._normalise_error(inner_exc)
                    warnings.warn(
                        f"Failed to invoke chat completions API for LLM rechecker: {inner_text}",
                        RuntimeWarning,
                    )
                    return None, f"LLM error: {inner_text}"
            if "max_tokens" in message:
                payload.pop("max_tokens", None)
                payload.setdefault("max_completion_tokens", self._max_output_tokens)
                try:
                    return completions.create(**payload), ""
                except Exception as inner_exc:  # pragma: no cover
                    inner_text = self._normalise_error(inner_exc)
                    warnings.warn(
                        f"Failed to invoke chat completions API for LLM rechecker: {inner_text}",
                        RuntimeWarning,
                    )
                    return None, f"LLM error: {inner_text}"
            warnings.warn(
                f"Failed to invoke chat completions API for LLM rechecker: {message}",
                RuntimeWarning,
            )
            return None, f"LLM error: {message}"

    @staticmethod
    def _normalise_error(exc: Exception) -> str:
        text = getattr(exc, "message", None) or str(exc)
        return text.strip() or exc.__class__.__name__

    @staticmethod
    def _interpret_reply(text: str) -> Tuple[Optional[bool], Optional[str]]:
        stripped = text.strip()
        if not stripped:
            return None, None
        verd_pattern = re.compile(r"^\s*verdict\s*[:=]\s*(yes|no|unknown)", re.IGNORECASE)
        reason_pattern = re.compile(r"reason\s*[:=]\s*(.+)", re.IGNORECASE | re.DOTALL)
        verdict_match = verd_pattern.search(stripped)
        verdict_value: Optional[bool]
        if verdict_match:
            token = verdict_match.group(1).lower()
            verdict_value = True if token == "yes" else False if token == "no" else None
            # try to locate a reason line after verdict
            reason_match = reason_pattern.search(stripped)
            rationale = reason_match.group(1).strip() if reason_match else None
            return verdict_value, rationale

        lowered = stripped.lower()
        for token, verdict in (("yes", True), ("no", False), ("unknown", None)):
            if lowered.startswith(token):
                remainder = stripped[len(token) :].lstrip(" :;,. -")
                rationale = remainder.strip() if remainder else None
                return verdict, rationale

        # Fallback: try to find an isolated verdict within the text
        upper = stripped.upper()
        if " YES " in f" {upper} ":
            return True, stripped
        if " NO " in f" {upper} ":
            return False, stripped
        if " UNKNOWN " in f" {upper} ":
            return None, stripped
        return None, stripped
