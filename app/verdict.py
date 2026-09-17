from collections.abc import Callable

from pydantic import BaseModel, Field, ValidationError, model_validator  # noqa: F401


class FailedRule(BaseModel):
    rule_id: int
    rule: str
    actual: str | float


class EligibilityVerdict(BaseModel):
    student_id: str = Field(pattern=r"^\d{2}[A-Z]{2}\d{3}$")
    drive_id: int
    eligible: bool
    failed_rules: list[FailedRule]
    summary: str = Field(min_length=1, max_length=280)

    @model_validator(mode="after")
    def rules_match_eligibility(self):
        if self.eligible and self.failed_rules:
            raise ValueError("eligible contradicts the presence of failed_rules")
        if not self.eligible and not self.failed_rules:
            raise ValueError("ineligible contradicts the absence of failed_rules")
        return self


class VerdictInvalid(Exception):
    def __init__(self, attempts: int, last_errors: list):
        super().__init__(f"no valid verdict after {attempts} attempts")
        self.attempts = attempts
        self.last_errors = last_errors


Generate = Callable[[list[str]], str]


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    return t.strip()


def structured_verdict(generate: Generate, prompt: str, max_retries: int = 2) -> EligibilityVerdict:
    """Ask the model for an EligibilityVerdict; feed validation errors back; give up after max_retries.

    TODO (stretch):
      - call generate(messages) with the conversation so far (starts as [prompt])
      - parse with EligibilityVerdict.model_validate_json(_strip_fences(raw))
      - on ValidationError: append the raw reply, then a feedback message containing
        "failed validation" and the errors, and try again
      - at most 1 + max_retries calls, then raise VerdictInvalid
    """
    messages = [prompt]
    errors = []
    for attempt in range(max_retries + 1):
        raw = generate(messages)
        try:
            return EligibilityVerdict.model_validate_json(_strip_fences(raw))
        except ValidationError as exc:
            errors = exc.errors()
            messages.extend([raw, f"Your response failed validation: {errors}"])
    raise VerdictInvalid(max_retries + 1, errors)
