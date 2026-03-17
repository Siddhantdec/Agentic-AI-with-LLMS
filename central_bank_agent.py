import uuid
from datetime import datetime
from typing import TypedDict

import streamlit as st

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END


# =========================================================
# STATE DEFINITION
# =========================================================

class ComplaintState(TypedDict):
    bank_id: str
    complaint_id: str
    complaint_text: str
    issue_category: str
    assigned_department: str
    status: str
    date_filed: str
    date_resolved: str
    resolution_summary: str
    regulatory_context: str


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def generate_id() -> str:
    return str(uuid.uuid4())[:8]

def today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


# =========================================================
# LLM SETUP (OPEN SOURCE)
# =========================================================

llm_pipeline = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_new_tokens=64
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

VALID_CATEGORIES = [
    "Fraud",
    "Payments",
    "Cybersecurity",
    "Compliance",
    "Supervision"
]


# =========================================================
# RAG SETUP (REGULATORY CIRCULARS)
# =========================================================

REGULATORY_DOCS = [
    "Cybersecurity incidents must be reported to the central bank within 6 hours.",
    "Fraud cases above the defined threshold require immediate escalation.",
    "Payment system failures affecting customers must be resolved within T+1.",
    "Compliance deviations must be reviewed by the supervision department."
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_texts(REGULATORY_DOCS, embeddings)
retriever = vectorstore.as_retriever()

MEMORY_DB = []


# =========================================================
# LANGGRAPH NODES
# =========================================================

def intake_node(state: ComplaintState):
    state["complaint_id"] = generate_id()
    state["date_filed"] = today()
    state["status"] = "Received"
    return state


def classify_issue(state: ComplaintState):
    prompt = (
        "Classify the banking issue into exactly ONE category:\n"
        "Fraud, Payments, Cybersecurity, Compliance, Supervision\n\n"
        f"Issue: {state['complaint_text']}\n\n"
        "Respond with only the category name."
    )

    response = llm.invoke(prompt).strip()
    state["issue_category"] = response if response in VALID_CATEGORIES else "Supervision"
    state["status"] = "Classified"
    return state


def rag_node(state: ComplaintState):
    docs = retriever.invoke(state["complaint_text"])
    state["regulatory_context"] = docs[0].page_content if docs else ""
    return state


def route_department(state: ComplaintState):
    mapping = {
        "Fraud": "Risk & Fraud Department",
        "Payments": "Payments & Settlement Department",
        "Cybersecurity": "Cyber Risk Department",
        "Compliance": "Compliance Department",
        "Supervision": "Supervision Department"
    }

    state["assigned_department"] = mapping[state["issue_category"]]
    state["status"] = "Assigned"
    return state


def resolve_issue(state: ComplaintState):
    state["resolution_summary"] = (
        f"Issue resolved by {state['assigned_department']} "
        f"using regulation: {state['regulatory_context']}"
    )
    state["date_resolved"] = today()
    state["status"] = "Resolved"
    return state


def memory_node(state: ComplaintState):
    MEMORY_DB.append(state.copy())
    return state


# =========================================================
# LANGGRAPH WORKFLOW
# =========================================================

graph = StateGraph(ComplaintState)

graph.add_node("intake", intake_node)
graph.add_node("classify", classify_issue)
graph.add_node("rag", rag_node)
graph.add_node("route", route_department)
graph.add_node("resolve", resolve_issue)
graph.add_node("memory", memory_node)

graph.set_entry_point("intake")

graph.add_edge("intake", "classify")
graph.add_edge("classify", "rag")
graph.add_edge("rag", "route")
graph.add_edge("route", "resolve")
graph.add_edge("resolve", "memory")
graph.add_edge("memory", END)

app = graph.compile()


# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(page_title="Central Bank Agent", layout="centered")

st.title("🏦 Central Bank Agentic Complaint System")
st.caption("LangGraph • LLM • RAG • Audit-Ready")

bank_id = st.text_input("Regional Bank ID", value="RB-01")
complaint_text = st.text_area("Describe the issue")

if st.button("Submit Complaint"):
    if not complaint_text.strip():
        st.warning("Please enter complaint details.")
    else:
        input_state: ComplaintState = {
            "bank_id": bank_id,
            "complaint_text": complaint_text,
            "complaint_id": "",
            "issue_category": "",
            "assigned_department": "",
            "status": "",
            "date_filed": "",
            "date_resolved": "",
            "resolution_summary": "",
            "regulatory_context": ""
        }

        result = app.invoke(input_state)

        st.success("Complaint processed successfully")

        st.markdown("### 📄 Complaint Details")
        st.write(f"**Complaint ID:** {result['complaint_id']}")
        st.write(f"**Category:** {result['issue_category']}")
        st.write(f"**Department:** {result['assigned_department']}")
        st.write(f"**Status:** {result['status']}")
        st.write(f"**Filed On:** {result['date_filed']}")
        st.write(f"**Resolved On:** {result['date_resolved']}")

        st.markdown("### 📘 Resolution")
        st.write(result["resolution_summary"])

with st.expander("🗄 Central Bank Audit Memory"):
    st.json(MEMORY_DB)
