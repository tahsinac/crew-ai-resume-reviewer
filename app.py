from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    WebsiteSearchTool
)
from tools.scraper_tools import ScraperTool
from PyPDF2 import PdfReader

def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

resume_text = get_pdf_text(r"resume\resume.pdf")
llm = ChatOpenAI(model="gpt-4-turbo-preview")
docs_tool = DirectoryReadTool(directory='./resume')
file_tool = FileReadTool()
web_rag_tool = WebsiteSearchTool()
scrape_tool = ScraperTool().scrape

analyzer_agent = Agent(
    role='ATS',
    goal='For a given job description, analyze a resume and see if it would be a good fit for the job.',
    backstory="""An advanced Application Tracking System (ATS) specializing in the tech industry, particularly in software engineering, data science, data analysis, and machine learning engineering. You can meticulously assess a given resume in relation to a provided job description.""",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    tools=[docs_tool, file_tool, scrape_tool]
)

job_link = 'https://jobs.lever.co/quincus/51f562da-f4ed-45f9-a8e2-4c56b02bae13'
analyze = Task(
    description=f'Analyze a resume for the job posted in the link {job_link}. The following is the resume: {resume_text}',
    expected_output='A score for the resume for the given job description. Higher the match between the resume and the job description, the better the score. Then give what key aspects are missing from the resume and advice on how the resume could be improved to be a good fit for the job. Use the entire job description you find in the link.',
    agent=analyzer_agent
)

# Set up your crew with a sequential process (tasks executed sequentially by default)
ats_crew = Crew(
    agents=[analyzer_agent],
    tasks=[analyze],
    # process=Process.hierarchical,
    manager_llm=llm,
)

# Initiate the crew to start working on its tasks
crew_result = ats_crew.kickoff()

print(crew_result)