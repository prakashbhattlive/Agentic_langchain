from typing import List
from pydantic import BaseModel, Field

class Source(BaseModel):
    """A schema for a source used by an agent."""

    url: str = Field(..., description="The URL of the source.")

class AgentResponse(BaseModel):
    """A schema for the response from an agent."""

    answer: str = Field(..., description="The main response from the agent.")
    sources: List[Source] = Field(
        default_factory=list,
        description="A list of sources used to generate the answer."
    )

