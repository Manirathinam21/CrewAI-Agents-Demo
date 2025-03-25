import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew    
from crewai_tools import SerperDevTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# API keys from the environment
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY') 

os.environ['GROQ_API_KEY'] = GROQ_API_KEY
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

search_tool= SerperDevTool()

def create_research_agent():
    #llm= ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    llm=ChatGroq(temperature=0,
             model_name="llama3-70b-8192",
             api_key=GROQ_API_KEY)

    return Agent(
        role= 'Research specialist',
        goal= 'Conduct through research on give topics',
        backstory= 'you are an experienced researcher with expertise in finding and synthesising information from various resource',
        verbose = True,
        allow_delegation=False,
        tools= [search_tool],
        llm= llm,
    )

def create_research_task(agent, topic):
    return Task(
        description= f'Research the following topic and provide a comprehensive summary: {topic}',
        agent=agent,
        expected_output= 'A detailed summary of the research findings, including key points and insights related to the topic'
    )

def run_research_agent(topic):
    agent= create_research_agent()
    task= create_research_task(agent, topic)
    crew= Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    return result

if __name__ == '__main__':
    print("Welcome to the Research Agent!")
    topic = input("Enter the research topic: ")
    result = run_research_agent(topic)
    print("Research Result:")
    print(result)