from typing import List
from pydantic import BaseModel,Field

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url:str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """Schema for the response with answer and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources:List[Source] = Field(default_factory=list,description="list of the sources used to answer")