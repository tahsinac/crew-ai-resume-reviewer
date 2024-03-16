from dotenv import load_dotenv
import streamlit as st
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

load_dotenv()
def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
# resume_text = get_pdf_text(r"resume\resume.pdf")

def get_resume_analysis(resume_text, job_link):
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

    if job_link.startswith('https://'):
        analyze = Task(
            description=f'Analyze a resume for the job posted in the link {job_link}. The following is the resume: {resume_text}',
            expected_output='A score for the resume for the given job description. Higher the match between the resume and the job description, the better the score. Also give a probability between 0 an 1 that this resume will pass ATS systems. Explain the reasoning behind the score and probability. Then give what key aspects are missing from the resume and advice on how the resume could be improved to be a good fit for the job. Use the entire job description you find in the link.',
            agent=analyzer_agent
        )
    else:
        analyze = Task(
            description=f'Analyze a resume for the following job description: ##{job_link}##. The following is the resume: ##{resume_text}##',
            expected_output='A score for the resume for the given job description. Higher the match between the resume and the job description, the better the score. Also give a probability between 0 an 1 that this resume will pass ATS systems. Explain the reasoning behind the score and probability. Then give what key aspects are missing from the resume and advice on how the resume could be improved to be a good fit for the job. Use the entire job description you find in the link.',
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

    return crew_result

st.title("ATS Analyzer")
job_link=st.text_area("Paste job URL or job description below", help="First Try with the URL. If the tool is unable to scrape the site, please paste the job discruiption directly.")
uploaded_file=st.file_uploader("Upload Your resume here",type="pdf",help="Upload your resume here in pdf format and press submit.")

submit = st.button("Submit")

if submit:
    if uploaded_file is not None:
        resume_text=get_pdf_text(uploaded_file)
        response=get_resume_analysis(resume_text, job_link)
        st.subheader("ANALYSIS")
        st.markdown(response)