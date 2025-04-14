from flask import Flask, render_template, request
import requests
import json
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os, sys



load_dotenv()


app = Flask('first_ai_agent')


# ---- CONFIGURATION ----
OPENAI_API_KEY =  os.getenv("OPENAI_API_KEY")
summary_cache = ""
extracted_cache = {}

# ---- PUBMED TOOL ----
# def search_pubmed(query: str, max_results: int = 1):
#     url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
#     params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": max_results}
#     response = requests.get(url, params=params)
#     id_list = response.json()["esearchresult"]["idlist"]
#     return id_list[0] if id_list else ""

def search_pubmed(query: str, max_results: int = 1):
    # Step 1: Search for the most recent PMID
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results
    }
    search_response = requests.get(search_url, params=search_params)
    id_list = search_response.json()["esearchresult"]["idlist"]

    if not id_list:
        return None, None

    pmid = id_list[0]

    # Step 2: Fetch the article title
    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    summary_params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "json"
    }
    summary_response = requests.get(summary_url, params=summary_params)
    title = summary_response.json()["result"][pmid]["title"]

    return pmid, title


def fetch_abstract(pmid: str):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pmid, "retmode": "text", "rettype": "abstract"}
    response = requests.get(url, params=params)
    return response.text

# ---- STRUCTURED PARSER ----
class ExtractedInfo(BaseModel):
    gene: str = Field(default="", description="Gene")
    disease: str = Field(default="", description="Disease")
    compound: str = Field(default="", description="Compound")

parser = PydanticOutputParser(pydantic_object=ExtractedInfo)

# ---- LLM SETUP ----
llm = ChatOpenAI(temperature=0.3, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")

# ---- PROMPTS & CHAINS ----
summarization_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    You are a biomedical expert. Read the following abstract and provide a detailed, informative summary suitable for a graduate-level reader. Include key findings, study methods, genes involved, conditions studied, and treatment outcomes if available.
    {text}
    """
)
summarize_chain = LLMChain(llm=llm, prompt=summarization_prompt)

def summarize(text: str) -> str:
    global summary_cache
    summary_cache = summarize_chain.run(text=text)
    return summary_cache

extraction_prompt = PromptTemplate(
    input_variables=["summary"],
    template="""
    You are an expert biomedical information extractor. Given the detailed summary below, extract the following information with specificity:
    
    - Gene(s) or genetic markers discussed
    - Disease or medical condition being studied
    - Compounds, drugs, or treatments mentioned
    
    Summary:
    {summary}
    
    {format_instructions}
    """,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
extract_chain = LLMChain(llm=llm, prompt=extraction_prompt)

def extract_info(summary: str) -> dict:
    global extracted_cache
    extraction = extract_chain.run(summary=summary)
    extracted_cache = parser.parse(extraction).model_dump()
    return extracted_cache

# def final_answer_tool(_: str = "") -> str:
#     return f"""
# <h3>Summary</h3>
# <p>{summary_cache}</p>

# <h3>Extracted Info</h3>
# <ul>
#     <li><strong>Gene:</strong> {extracted_cache.get('gene', 'Not mentioned')}</li>
#     <li><strong>Disease:</strong> {extracted_cache.get('disease', 'Not mentioned')}</li>
#     <li><strong>Compound:</strong> {extracted_cache.get('compound', 'Not mentioned')}</li>
# </ul>

# <h3>Insight</h3>
# <p>This study explores the relationship between <strong>{extracted_cache.get('gene')}</strong> and <strong>{extracted_cache.get('disease')}</strong>, and evaluates the therapeutic use of <strong>{extracted_cache.get('compound')}</strong>.</p>
# """

# ---- ROUTE ----
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        topic = request.form["topic"]
        pmid, title = search_pubmed(topic)
        
        if not pmid:
            result = "<p>No papers found.</p>"
        else:
            abstract = fetch_abstract(pmid)
            summary = summarize(abstract)
            extract_info(summary)

            # Create PubMed link
            pubmed_link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            # Build result with title at the top
            result = f"""
            <h2>📝 Paper Title</h2>
            <p><strong><a href="{pubmed_link}" target="_blank">{title}</a></strong></p>

            <h3>📄 Summary</h3>
            <p>{summary_cache}</p>

            <h3>🧬 Extracted Info</h3>
            <ul>
                <li><strong>Gene:</strong> {extracted_cache.get('gene', 'Not mentioned')}</li>
                <li><strong>Disease:</strong> {extracted_cache.get('disease', 'Not mentioned')}</li>
                <li><strong>Compound:</strong> {extracted_cache.get('compound', 'Not mentioned')}</li>
            </ul>

            <h3>🧠 Insight</h3>
            <p>This study explores the relationship between <strong>{extracted_cache.get('gene')}</strong> and <strong>{extracted_cache.get('disease')}</strong>, and evaluates the therapeutic use of <strong>{extracted_cache.get('compound')}</strong>.</p>
            """

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
