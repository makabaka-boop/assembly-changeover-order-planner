"""Request / response schemas with strict validation.

Any malformed request (wrong types, extra/unknown fields, empty or non-ASCII
ids/families, wrong cardinality) is rejected with HTTP 422 by FastAPI's
validation layer.  Semantic graph checks (unknown edge endpoints, self loops,
duplicate edges, duplicate job ids) are checked explicitly in the endpoint and
also produce 422.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


def _ascii_only(value: str) -> str:
    if not value.isascii():
        raise ValueError("must contain ASCII characters only")
    return value


# Non-empty ASCII identifier.  min_length is enforced via Field on each usage.
AsciiId = Annotated[str, AfterValidator(_ascii_only)]


class JobIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: AsciiId = Field(min_length=1)
    family: AsciiId = Field(min_length=1)


class EdgeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    before: AsciiId = Field(min_length=1)
    after: AsciiId = Field(min_length=1)


class ScheduleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jobs: list[JobIn] = Field(min_length=2, max_length=18)
    edges: list[EdgeIn] = Field(default_factory=list, max_length=100)


class ChangeoverOut(BaseModel):
    position: int
    job_id: str
    from_family: str
    to_family: str


class ScheduleResponse(BaseModel):
    status: Literal["OK"]
    order: list[str]
    changeover_count: int
    changeovers: list[ChangeoverOut]


class CycleResponse(BaseModel):
    status: Literal["CYCLE"]
    order: list[str]
    changeover_count: int
    changeovers: list[ChangeoverOut]
    cycle: list[str]


ScheduleAPIResponse = Annotated[
    Union[ScheduleResponse, CycleResponse],
    Field(discriminator="status"),
]
