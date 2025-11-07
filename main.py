from __future__ import annotations

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from schemas import AgentResponse

load_dotenv()

SYSTEM_PROMPT = (
    "You are a job search assistant. Use the available tools to find up-to-date "
    "information about AI engineer roles that mention LangChain. Always populate the "
    "structured response schema and cite the source URLs you rely on."
)

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o-mini")

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    response_format=AgentResponse,
)


def run_job_search(query: str) -> AgentResponse:
    """Invoke the agent and coerce the result into AgentResponse."""
    result_state = agent.invoke({"messages": [HumanMessage(content=query)]})

    structured = result_state.get("structured_response")
    if structured is not None:
        if isinstance(structured, AgentResponse):
            return structured
        return AgentResponse.model_validate(structured)

    # Fallback: return the latest AI message if structured output is missing.
    for message in reversed(result_state["messages"]):
        if isinstance(message, AIMessage):
            content = message.content
            answer = content if isinstance(content, str) else str(content)
            return AgentResponse(answer=answer, sources=[])

    return AgentResponse(answer="No answer generated.", sources=[])


def main():
    print("Hello from langchain-course!")
    query = (
        "Search for 3 job postings for an AI engineer using LangChain in the Bay Area "
        "on LinkedIn and list their details."
    )
    result = run_job_search(query)
    print(result)


if __name__ == "__main__":
    main()
