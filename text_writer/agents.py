import functools
import operator

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from typing_extensions import TypedDict
from typing import Annotated, List, Dict, Optional
from utilities import *
from tools import *
from IPython.display import Image

# GenerationTeam graph state
llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0.8)

class GenerationTeamState(TypedDict):
    # A message is added after each team member finishes
    messages: Annotated[List[BaseMessage], operator.add]
    # The team members are tracked so they are aware of
    # the others' skill-sets
    team_members: List[str]
    # Used to route work. The supervisor calls a function
    # that will update this every time it makes a decision
    next: str

search_agent = create_agent(
    llm,
    [google_search],
    "You are a research assistant who can search for up-to-date info using the google search engine.",
)
search_node = functools.partial(agent_node, agent=search_agent, name="Search")

research_agent = create_agent(
    llm,
    [scrape_webpages],
    "You are a research assistant who can scrape specified urls for more detailed information using the scrape_webpages function.",
)
research_node = functools.partial(agent_node, agent=research_agent, name="WebScraper")

references_agent = create_agent(
    llm,
    [scrape_references],
    "You are an agent focused on scraping predefined reference links using the scrape_webpages function.",
)
references_node = functools.partial(agent_node, agent=references_agent, name="ReferencesScraper")

translate_agent = create_agent(
    llm,
    [google_translate],
    "You are a research assistant who can translate text from english to russian.",
)
translate_node = functools.partial(agent_node, agent=translate_agent, name="Translator")

supervisor_agent = create_team_supervisor(
    llm,
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: WebScraper, ReferencesScraper, Translator "
    " Given the following user request, respond with the worker to act next. "
    " Each worker will perform a task and respond with their results and status. When finished,"
    " respond with FINISH.",
    ["WebScraper", "ReferencesScraper", "Translator", "Search"],
)


# Compile research team
research_graph = StateGraph(GenerationTeamState)
research_graph.add_node("WebScraper", research_node)
research_graph.add_node("Search", search_node)
research_graph.add_node("ReferencesScraper", references_node)
research_graph.add_node("Translator", translate_node)
research_graph.add_node("supervisor", supervisor_agent)

research_graph.add_edge("WebScraper", "supervisor")
research_graph.add_edge("Search", "supervisor")
research_graph.add_edge("ReferencesScraper", "supervisor")
research_graph.add_edge("Translator", "supervisor")
research_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {"WebScraper": "WebScraper", 
     "Search": "Search", 
     "ReferencesScraper": "ReferencesScraper", 
     "Translator": "Translator", 
     "FINISH": END},
)

research_graph.add_edge(START, "supervisor")
chain = research_graph.compile()

def enter_chain(message: str):
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results

research_chain = enter_chain | chain

import operator
from pathlib import Path


# Document writing team graph state
class DocWritingState(TypedDict):
    # This tracks the team's conversation internally
    messages: Annotated[List[BaseMessage], operator.add]
    # This provides each worker with context on the others' skill sets
    team_members: str
    # This is how the supervisor tells langgraph who to work next
    next: str
    # This tracks the shared directory state
    current_files: str

def prelude(state):
    written_files = []
    if not WORKING_DIRECTORY.exists():
        WORKING_DIRECTORY.mkdir()
    try:
        written_files = [
            f.relative_to(WORKING_DIRECTORY) for f in WORKING_DIRECTORY.rglob("*")
        ]
    except Exception:
        pass
    if not written_files:
        return {**state, "current_files": "No files written."}
    return {
        **state,
        "current_files": "\nBelow are files your team has written to the directory:\n"
        + "\n".join([f" - {f}" for f in written_files]),
    }


doc_writer_agent = create_agent(
    llm,
    [write_document, edit_document, read_document],
    "You are an expert writing a research document.\n"
    # The {current_files} value is populated automatically by the graph state
    "Below are files currently in your directory:\n{current_files}",
)
# Injects current directory working state before each call
context_aware_doc_writer_agent = prelude | doc_writer_agent
doc_writing_node = functools.partial(
    agent_node, agent=context_aware_doc_writer_agent, name="DocWriter"
)

note_taking_agent = create_agent(
    llm,
    [create_outline, read_document],
    "You are an expert senior researcher tasked with writing a paper outline and"
    " taking notes to craft a perfect paper.{current_files}",
)
context_aware_note_taking_agent = prelude | note_taking_agent
note_taking_node = functools.partial(
    agent_node, agent=context_aware_note_taking_agent, name="NoteTaker"
)

doc_writing_supervisor = create_team_supervisor(
    llm,
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {team_members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.",
    ["DocWriter", "NoteTaker"],
)

# Create the graph here:
# Note that we have unrolled the loop for the sake of this doc
authoring_graph = StateGraph(DocWritingState)
authoring_graph.add_node("DocWriter", doc_writing_node)
authoring_graph.add_node("NoteTaker", note_taking_node)
authoring_graph.add_node("supervisor", doc_writing_supervisor)

# Add the edges that always occur
authoring_graph.add_edge("DocWriter", "supervisor")
authoring_graph.add_edge("NoteTaker", "supervisor")

# Add the edges where routing applies
authoring_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "DocWriter": "DocWriter",
        "NoteTaker": "NoteTaker",
        "FINISH": END,
    },
)

authoring_graph.add_edge(START, "supervisor")
chain = authoring_graph.compile()

def enter_chain(message: str, members: List[str]):
    results = {
        "messages": [HumanMessage(content=message)],
        "team_members": ", ".join(members),
    }
    return results

authoring_chain = (
    functools.partial(enter_chain, members=authoring_graph.nodes)
    | authoring_graph.compile()
)

supervisor_node = create_team_supervisor(
    llm,
    "You are a supervisor tasked with managing a conversation between the"
    " following teams: {team_members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.",
    ["ResearchTeam", "PaperWritingTeam"],
)

# Top-level graph state
class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str


def get_last_message(state: State) -> str:
    return state["messages"][-1].content


def join_graph(response: dict):
    return {"messages": [response["messages"][-1]]}


# Define the graph.
super_graph = StateGraph(State)
# First add the nodes, which will do the work
super_graph.add_node("ResearchTeam", get_last_message | research_chain | join_graph)
super_graph.add_node(
    "PaperWritingTeam", get_last_message | authoring_chain | join_graph
)
super_graph.add_node("supervisor", supervisor_node)

# Define the graph connections, which controls how the logic
# propagates through the program
super_graph.add_edge("ResearchTeam", "supervisor")
super_graph.add_edge("PaperWritingTeam", "supervisor")
super_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "PaperWritingTeam": "PaperWritingTeam",
        "ResearchTeam": "ResearchTeam",
        "FINISH": END,
    },
)
super_graph.add_edge(START, "supervisor")
super_graph = super_graph.compile()

for s in super_graph.stream(
    {
        "messages": [
            HumanMessage(
                content="The main goal of your group is to write a popular science paper for young enterpreneurs, "
                "explaining them how they can create their own site. Treat the paragraphs one by one separately! Do not generate whole text in one run! "
                "and write them separately,\ to the current directory. After each write send the link to written text. Also, use agent for references on each bullet point! "
                " You need to mention all of the ollowing paragraphs: "
                "1) What domain is? How it works? How you can register an mail address for corporate account?"
                "2) What is whois service? How it works?"
                "3) What is hosting? What are the differences between hosting and VDS/VPS? Where you can host simple sites, and where — large projects?"
                "4) What is IP address?"
                "5) What is a site constructor? What is the differences between CMS and site constructor?"
                "6) What are the differences between cloud server and hosting? Why a lot of people use cloud nowadays?"
                "7) What is SEO optimization and Targeted PR, and how people ust it? "
                "8) What is an internet acquiring, and how you can integrate it to your site? "
                "9) What is a DDoS, and what are the main vulnerabilities on the site? "
                "10) What is an SSL certificate, and why are they necessary? Who gives this certificates? "
                "11) What are datacenters? Which tiers do they have? "
                "12) What is uptime? Why not all of the uptimes are fair? "
                "The total length of the whole text should be ~11000 symbols. "
            )
        ],
    },
    {"recursion_limit": 150},
):
    if "__end__" not in s:
        print(s)
        print("---")

