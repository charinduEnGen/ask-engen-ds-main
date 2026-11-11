from typing import Any, Literal, Optional

from pydantic import AliasChoices, BaseModel, Field, field_validator


class UserRequest(BaseModel):
    """Incoming DS-endpoint request. `mode` carries the project name end to
    end from the backend (see POST /query/{project})."""

    user_id: str
    session_id: str
    question_id: Optional[str] = None
    # The real caller sends JSON null (not "") when this is a reply to a
    # merchant-disambiguation prompt — the frontend has no question text to
    # send, it's recovered server-side from the pending disambiguation
    # record (see query_orchestrator._resolve_pending_disambiguation).
    # Normalized to "" below so every str-typed consumer downstream (the
    # whole pipeline expects `query: str`) never has to special-case None.
    query: str
    mode: str = "DEFAULT"
    answerId: Optional[str] = Field(default=None, max_length=128)
    # Selected option ids from a prior turn's MerchantDisambiguation prompt
    # (QueryResponse.merchantDisambiguation.options[].id), e.g. ["carnage-1",
    # "carnage-3"]. Only meaningful together with `answerId` naming the turn
    # that asked the question — that's how the pipeline looks back up which
    # question was pending and which merchant_name values the ids resolve to.
    #
    # The real caller sends this as either a bare `false` (not a
    # disambiguation reply) or `{"enabled": true, "values": [...]}` (a
    # reply) — never a flat list directly. `merchantDisambiguation` (also
    # accepted, for internal callers/tests) is kept as the plain
    # `Optional[list[str]]` shape every other module in this pipeline
    # already reads; _normalize_disambiguation_shape below is what
    # translates the real wire shape into it.
    merchantDisambiguation: Optional[list[str]] = Field(
        default=None,
        validation_alias=AliasChoices("merchantDisambiguation", "merchant_Disambiguation"),
    )

    @field_validator("query", mode="before")
    @classmethod
    def _normalize_null_query(cls, value: Any) -> str:
        return "" if value is None else value

    @field_validator("answerId", mode="before")
    @classmethod
    def _normalize_null_string(cls, value: Any) -> Optional[str]:
        # The DS payload sends the literal string "null" instead of JSON null.
        if isinstance(value, str) and value.strip().lower() == "null":
            return None
        return value

    @field_validator("merchantDisambiguation", mode="before")
    @classmethod
    def _normalize_disambiguation_shape(cls, value: Any) -> Optional[list[str]]:
        # Real wire shapes: bare `false` (no reply), or
        # {"enabled": bool, "values": [...]} (a reply, values = selected
        # option ids, occasionally with stray whitespace). A plain list is
        # also accepted as-is for internal callers/tests that already speak
        # the resolved shape directly.
        if value is None or value is False:
            return None
        if isinstance(value, bool):
            # Bare `true` with no values payload shouldn't happen from the
            # real caller, but degrade to "nothing selected" rather than
            # erroring the request.
            return None
        if isinstance(value, dict):
            if not value.get("enabled"):
                return None
            values = value.get("values") or []
        elif isinstance(value, list):
            values = value
        else:
            return None
        cleaned = [str(v).strip() for v in values if str(v).strip()]
        return cleaned or None


class ResponseMeta(BaseModel):
    mode: str
    dataRefreshedAt: str
    generatedAt: str


class MerchantDisambiguationOption(BaseModel):
    id: str
    label: str


class MerchantDisambiguation(BaseModel):
    question: str
    selectionType: str
    options: list[MerchantDisambiguationOption]


class AnswerBlock(BaseModel):
    """The three-form answer: text alone, text+table, or text+table+chart.

    `table` and `chart` shapes vary (flat vs paginated table; pie/bar/line
    nest differently), so they stay as plain dicts built by
    src.tools.response_builder rather than rigid sub-models.
    """

    text: dict[str, Any]
    table: Optional[dict[str, Any]] = None
    chart: Optional[dict[str, Any]] = None


class QueryResponse(BaseModel):
    answerId: str
    questionId: str
    userId: str
    sessionId: str
    meta: ResponseMeta
    maxContextWindow: bool = False
    chatExpire: bool = False
    merchantDisambiguation: Optional[MerchantDisambiguation] = None
    answer: AnswerBlock
    followUps: Optional[list[str]] = None
