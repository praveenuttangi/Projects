from flask import Flask, render_template, request
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv
import os
load_dotenv()


# Setup
app = Flask("second_app")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

# Define state
class PatientCase(TypedDict):
    radiologist_note: str
    clinical_note: str
    radiology_output: str
    clinical_output: str
    fused_case: str
    treatment_plan: str
    patient_summary: str

llm = ChatOpenAI(
    temperature=0.3,
    model_name="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY
)

def call_openai(prompt, system="You are a helpful medical assistant."):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    return llm.invoke(messages).content

# Agents
def radiology_agent(state: PatientCase) -> PatientCase:
    prompt = f"""
    Extract structured insights from this radiology report:
    - Tumor size and location
    - Impression (benign/malignant)
    - Recommended action

    Report:
    {state['radiologist_note']}
    """
    state["radiology_output"] = call_openai(prompt, "You are a radiology AI assistant.")
    return state

def clinical_agent(state: PatientCase) -> PatientCase:
    prompt = f"""
    Extract patient details from the clinical note:
    - Age, sex, smoking history
    - Symptoms
    - Genetic test results (EGFR, ALK, PD-L1)

    Note:
    {state['clinical_note']}
    """
    state["clinical_output"] = call_openai(prompt, "You are a clinical data analyst.")
    return state

def fusion_agent(state: PatientCase) -> PatientCase:
    prompt = f"""
    Combine radiology and clinical findings into a structured summary for oncology decision-making.

    Radiology:
    {state['radiology_output']}

    Clinical:
    {state['clinical_output']}
    """
    state["fused_case"] = call_openai(prompt, "You are a clinical reasoning AI.")
    return state

def treatment_agent(state: PatientCase) -> PatientCase:
    prompt = f"""
    Based on the following case summary, recommend:
    - First-line treatment
    - Targeted therapy (if applicable)
    - Justification

    Case Summary:
    {state['fused_case']}
    """
    state["treatment_plan"] = call_openai(prompt, "You are an AI oncology assistant.")
    return state

def report_agent(state: PatientCase) -> PatientCase:
    prompt = f"""
    Convert the treatment plan into a simple explanation for the patient:

    Treatment Plan:
    {state['treatment_plan']}
    """
    state["patient_summary"] = call_openai(prompt, "You are a patient education AI.")
    return state

# Build LangGraph
graph = StateGraph(PatientCase)
graph.add_node("Radiology", radiology_agent)
graph.add_node("Clinical", clinical_agent)
graph.add_node("Fusion", fusion_agent)
graph.add_node("Treatment", treatment_agent)
graph.add_node("Summary", report_agent)
graph.set_entry_point("Radiology")
graph.add_edge("Radiology", "Clinical")
graph.add_edge("Clinical", "Fusion")
graph.add_edge("Fusion", "Treatment")
graph.add_edge("Treatment", "Summary")
graph.add_edge("Summary", END)

cancer_graph = graph.compile()

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    output = None
    if request.method == "POST":
        rad_note = request.form.get("radiologist_note")
        clinical_note = request.form.get("clinical_note")
        
        inputs = {
            "radiologist_note": rad_note,
            "clinical_note": clinical_note,
            "radiology_output": "",
            "clinical_output": "",
            "fused_case": "",
            "treatment_plan": "",
            "patient_summary": ""
        }

        result = cancer_graph.invoke(inputs)
        output = {
            "Radiology Insight": result["radiology_output"],
            "Clinical Insight": result["clinical_output"],
            "Fused Summary": result["fused_case"],
            "Treatment Plan": result["treatment_plan"],
            "Patient Summary": result["patient_summary"]
        }

    return render_template("index.html", output=output)

if __name__ == "__main__":
    app.run(debug=True)
