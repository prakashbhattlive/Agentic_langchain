from dotenv import load_dotenv
load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
#from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
#llm = ChatOllama(model="gpt-oss:20b", base_url="http://192.168.1.6:11434", temperature=0)

react_prompt = hub.pull("hwchase17/react")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)

REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tool_names"],
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
).partial(format_instructions=output_parser.get_format_instructions())

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
    )
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(
    lambda x: x["output"]
)
parse_output = RunnableLambda(
    lambda x: output_parser.parse(x)
)

chain = agent_executor | extract_output | parse_output


def main():
    result = agent_executor.invoke(
        {
            "input": "search for the 3 jobs posting for SRE with AI using langchain in Pune, India"
        }
    )

    print(result)

if __name__ == "__main__":
    main()
