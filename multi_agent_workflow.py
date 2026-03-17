# ================================================================
# PRACTICAL MULTI-AGENT WORKFLOW
# Use Case: "Company Research Assistant"
# ================================================================
#
# REAL-WORLD SCENARIO:
#   A sales/business team member types:
#   "Research Tesla and prepare a briefing report + email draft"
#
#   The system automatically:
#   1. RESEARCHER  → searches the web for company info
#   2. ANALYST     → queries internal sales database (SQL)
#   3. SUMMARIZER  → combines findings into a clean report
#   4. EMAIL AGENT → drafts a professional email
#   5. GUARDRAILS  → safety check + rate limit + audit log
#
# WHY THIS IS INDUSTRY-RELEVANT:
#   - Sales teams use this before client meetings
#   - Analysts use this for due diligence
#   - Marketing uses this for competitor research
#   - Consulting firms build exactly this for clients
#
# TECH STACK:
#   - Groq (free, fast LLM)
#   - LangChain (agent framework)
#   - SQLite (internal database simulation)
#   - DuckDuckGo (free web search)
#   - python-dotenv (secure key management)
#
# INSTALL:
#   pip install langchain==0.2.16 langchain-core==0.2.38
#          langchain-groq==0.1.9 langchain-community==0.2.16
#          ddgs requests python-dotenv
#
# SETUP:
#   Create .env file → GROQ_API_KEY=gsk_xxxxxxxxxxxx
#   Get free key at: https://console.groq.com
# ================================================================

import os
import re
import json
import time
import sqlite3
import logging
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate

# Web search
from ddgs import DDGS

load_dotenv()

# ================================================================
# STEP 0: CONFIGURATION
# ================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "gsk_your_key_here")
MODEL_NAME   = "llama-3.1-8b-instant"   # Fast + free on Groq

print("=" * 60)
print("  COMPANY RESEARCH ASSISTANT — Multi-Agent Workflow")
print("=" * 60)


# ================================================================
# STEP 1: SETUP THE DATABASE (Simulates company's internal CRM)
# ================================================================
# In real industry: this would be Salesforce, HubSpot, or a
# company's own PostgreSQL/MySQL database
# Here we use SQLite (in-memory) to keep it simple

def create_internal_database():
    """
    Creates a simple in-memory SQLite database simulating
    a company's internal CRM (Customer Relationship Management) system.
    """
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Table 1: Companies we've interacted with
    cursor.execute("""
        CREATE TABLE companies (
            id          INTEGER PRIMARY KEY,
            name        TEXT,
            industry    TEXT,
            revenue_usd REAL,
            employees   INTEGER,
            status      TEXT,
            last_contact TEXT
        )
    """)

    # Table 2: Our past deals with these companies
    cursor.execute("""
        CREATE TABLE deals (
            id          INTEGER PRIMARY KEY,
            company_id  INTEGER,
            deal_value  REAL,
            stage       TEXT,
            product     TEXT,
            close_date  TEXT
        )
    """)

    # Insert sample company data
    cursor.executemany("""
        INSERT INTO companies VALUES (?,?,?,?,?,?,?)
    """, [
        (1, "Tesla",        "Electric Vehicles",   81500000000, 127855, "Active Customer",  "2024-11-01"),
        (2, "Apple",        "Technology",          383000000000, 161000, "Prospect",        "2024-10-15"),
        (3, "Microsoft",    "Technology",          211900000000, 221000, "Active Customer",  "2024-12-01"),
        (4, "Amazon",       "E-Commerce/Cloud",    574800000000, 1540000,"Cold Lead",        "2024-09-20"),
        (5, "Netflix",      "Entertainment",        33700000000,  13000, "Prospect",         "2024-11-20"),
    ])

    # Insert sample deals data
    cursor.executemany("""
        INSERT INTO deals VALUES (?,?,?,?,?,?)
    """, [
        (1, 1,  250000, "Closed Won",  "AI Analytics Platform", "2024-06-15"),
        (2, 1,  180000, "Negotiation", "Data Pipeline Tool",    "2025-01-30"),
        (3, 3,  450000, "Closed Won",  "AI Analytics Platform", "2024-08-01"),
        (4, 3,  320000, "Proposal",    "Security Suite",        "2025-02-15"),
        (5, 5,   95000, "Discovery",   "AI Analytics Platform", "2025-03-01"),
    ])

    conn.commit()
    print("✅ Internal database created (CRM simulation)")
    return conn

# Create the database — shared across all agents
DB_CONNECTION = create_internal_database()


# ================================================================
# STEP 2: DEFINE THE TOOLS
# Each tool is a Python function decorated with @tool
# Agents will CHOOSE which tools to use based on the task
# ================================================================

# ── Tool 1: Web Search ───────────────────────────────────────
@tool
def search_company_news(company_name: str) -> str:
    """
    Search the internet for latest news and information about a company.
    Use this tool when you need current, real-world information about
    a company's products, strategy, financials, or recent events.

    Input: company name (e.g., 'Tesla', 'Apple Inc')
    Output: news articles and key facts about the company
    """
    print(f"\n  🔍 [Web Search] Searching for: {company_name}")
    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                f"{company_name} company news 2024 2025",
                max_results=4
            )
            if not results:
                return f"No news found for {company_name}"

            # Format results clearly
            formatted = f"WEB SEARCH RESULTS FOR: {company_name}\n"
            formatted += "=" * 40 + "\n"
            for i, r in enumerate(results, 1):
                formatted += f"\n[Article {i}] {r['title']}\n"
                formatted += f"  {r['body'][:300]}...\n"
                formatted += f"  Source: {r['href']}\n"
            return formatted

    except Exception as e:
        return f"Search failed: {str(e)}. Using placeholder data for demo."


# ── Tool 2: Internal Database Query (SQL) ────────────────────
@tool
def query_internal_crm(company_name: str) -> str:
    """
    Query the internal company CRM database to find information about
    our relationship with a company — past deals, revenue, contact history.
    Use this when you need OUR internal data about a company we work with.

    Input: company name (e.g., 'Tesla', 'Microsoft')
    Output: Internal CRM data — deals, status, revenue info
    """
    print(f"\n  🗄️  [SQL Tool] Querying CRM for: {company_name}")
    try:
        cursor = DB_CONNECTION.cursor()

        # Query 1: Company info
        cursor.execute("""
            SELECT name, industry, revenue_usd, employees, status, last_contact
            FROM companies
            WHERE LOWER(name) LIKE LOWER(?)
        """, (f"%{company_name}%",))
        company_row = cursor.fetchone()

        if not company_row:
            return f"No internal records found for '{company_name}'. This may be a new prospect."

        company_id = cursor.execute(
            "SELECT id FROM companies WHERE LOWER(name) LIKE LOWER(?)",
            (f"%{company_name}%",)
        ).fetchone()[0]

        # Query 2: Past deals
        cursor.execute("""
            SELECT deal_value, stage, product, close_date
            FROM deals
            WHERE company_id = ?
            ORDER BY deal_value DESC
        """, (company_id,))
        deals = cursor.fetchall()

        # Format the output
        output = f"INTERNAL CRM DATA FOR: {company_name}\n"
        output += "=" * 40 + "\n"
        output += f"Industry:      {company_row[1]}\n"
        output += f"Revenue:       ${company_row[2]:,.0f}\n"
        output += f"Employees:     {company_row[3]:,}\n"
        output += f"Status:        {company_row[4]}\n"
        output += f"Last Contact:  {company_row[5]}\n"

        if deals:
            output += f"\nDEAL HISTORY ({len(deals)} deals):\n"
            total = 0
            for deal in deals:
                output += f"  • ${deal[0]:,.0f} | {deal[2]} | Stage: {deal[1]} | {deal[3]}\n"
                if deal[1] == "Closed Won":
                    total += deal[0]
            output += f"\nTotal Closed Revenue: ${total:,.0f}\n"
        else:
            output += "\nNo past deals found.\n"

        return output

    except Exception as e:
        return f"Database query failed: {str(e)}"


# ── Tool 3: Calculate Business Metrics ───────────────────────
@tool
def calculate_business_metrics(expression: str) -> str:
    """
    Calculate business metrics like deal potential, growth rates,
    market share percentages, or revenue projections.
    Use this for any math calculations needed in the analysis.

    Input: a Python math expression (e.g., '250000 * 1.2' or '450000 / 3')
    Output: the calculated result
    """
    print(f"\n  🧮 [Calculator] Computing: {expression}")
    try:
        # Safe eval — only allows math operations
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result:,.2f}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


# ── Tool 4: Draft Email ───────────────────────────────────────
@tool
def draft_professional_email(context: str) -> str:
    """
    Draft a professional business email based on the research context.
    Use this LAST, after you have gathered all company information.
    The context should include company name, key facts, and purpose of email.

    Input: summary of company info and email purpose
    Output: a complete professional email draft
    """
    print(f"\n  ✉️  [Email Tool] Drafting email...")

    llm = ChatGroq(
        model=MODEL_NAME,
        groq_api_key=GROQ_API_KEY,
        temperature=0.4,
        streaming=False,
    )

    email_prompt = f"""Write a professional business email based on this context:

{context}

Format EXACTLY as:
Subject: [subject line]

Dear [Name],

[Opening paragraph - 2 sentences]

[Main body - 3-4 sentences with key points]

[Call to action - 1-2 sentences]

Best regards,
[Your Name]
Business Development Team
"""
    response = llm.invoke([HumanMessage(content=email_prompt)])
    return f"\n{'='*40}\nEMAIL DRAFT:\n{'='*40}\n{response.content}"


# Collect all tools in one list
ALL_TOOLS = [
    search_company_news,
    query_internal_crm,
    calculate_business_metrics,
    draft_professional_email
]


# ================================================================
# STEP 3: GUARDRAILS
# These run BEFORE any agent processes a request
# Industry standard: every production system has these
# ================================================================

class Guardrails:
    """
    Safety and control mechanisms for the agent system.

    In industry, these prevent:
    - Misuse (asking the agent to do harmful things)
    - Runaway costs (too many API calls)
    - Compliance issues (GDPR, data security)
    - Debugging nightmares (no audit trail)
    """

    # Words that should never be in requests
    BLOCKED_TERMS = [
        "password", "hack", "exploit", "delete all",
        "drop table", "credit card", "ssn", "illegal"
    ]

    # Rate limiting: max requests per minute
    MAX_REQUESTS_PER_MINUTE = 10
    _request_times = []

    # Audit log: records every request
    _audit_log = []

    @classmethod
    def safety_check(cls, user_input: str) -> tuple:
        """
        Check if the request is safe to process.
        Returns: (is_safe: bool, reason: str)
        """
        lower_input = user_input.lower()
        for term in cls.BLOCKED_TERMS:
            if term in lower_input:
                return False, f"Request blocked: contains restricted term '{term}'"
        return True, "OK"

    @classmethod
    def rate_limit_check(cls) -> tuple:
        """
        Check if we're within rate limits.
        Returns: (is_allowed: bool, message: str)
        """
        now = time.time()
        # Keep only requests from last 60 seconds
        cls._request_times = [t for t in cls._request_times if now - t < 60]

        if len(cls._request_times) >= cls.MAX_REQUESTS_PER_MINUTE:
            wait = 60 - (now - cls._request_times[0])
            return False, f"Rate limit reached. Please wait {wait:.0f} seconds."

        cls._request_times.append(now)
        return True, "OK"

    @classmethod
    def human_in_the_loop(cls, action: str, details: str,
                           auto_approve: bool = True) -> bool:
        """
        For HIGH-RISK actions (sending emails, deleting data),
        pause and ask a human to approve.

        auto_approve=True  → for demo (skips the prompt)
        auto_approve=False → for production (requires yes/no input)
        """
        risky_actions = ["send_email", "delete", "post", "publish"]
        is_risky = any(r in action.lower() for r in risky_actions)

        if not is_risky:
            return True  # Low-risk — auto approve

        if auto_approve:
            print(f"\n  👤 [HITL] Action '{action}' auto-approved (demo mode)")
            return True

        # In production: send Slack message / email to manager
        print(f"\n  ⚠️  HUMAN APPROVAL REQUIRED")
        print(f"  Action: {action}")
        print(f"  Details: {details[:100]}...")
        approval = input("  Approve? (yes/no): ").strip().lower()
        return approval in ("yes", "y")

    @classmethod
    def audit_log(cls, event: str, input_data: str,
                  output: str = "", status: str = "success"):
        """
        Record every action for compliance and debugging.
        In production: write to a database or cloud logging service.
        """
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event":     event,
            "input":     input_data[:150],
            "output":    output[:150],
            "status":    status
        }
        cls._audit_log.append(entry)

        # Also write to a file (like production systems do)
        with open("audit_log.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

    @classmethod
    def print_audit_summary(cls):
        """Show a summary of all logged actions."""
        print("\n" + "─" * 60)
        print("  📋 AUDIT LOG SUMMARY")
        print("─" * 60)
        total   = len(cls._audit_log)
        success = sum(1 for e in cls._audit_log if e["status"] == "success")
        blocked = sum(1 for e in cls._audit_log if e["status"] == "blocked")
        print(f"  Total actions  : {total}")
        print(f"  Successful     : {success}")
        print(f"  Blocked        : {blocked}")
        print(f"  Audit file     : audit_log.jsonl")
        for entry in cls._audit_log[-3:]:  # Show last 3 entries
            print(f"\n  [{entry['timestamp']}] {entry['event']} → {entry['status']}")
            print(f"    Input: {entry['input'][:80]}...")


# ================================================================
# STEP 4: CREATE THE LLM (Brain of the Agent)
# ================================================================

def create_llm():
    """
    Create the LLM that powers all agents.
    Using Groq + Llama 3.1 8B = fast, free, capable.
    """
    return ChatGroq(
        model=MODEL_NAME,
        groq_api_key=GROQ_API_KEY,
        temperature=0,       # 0 = deterministic (good for business tasks)
        streaming=False,     # must be False for ReAct agents on Groq
    )


# ================================================================
# STEP 5: CREATE THE REACT AGENT
# ReAct = Reason + Act (the most common agent pattern)
# The agent thinks → picks a tool → uses it → thinks again
# ================================================================

def create_company_research_agent():
    """
    Create a ReAct agent that uses our 4 tools.

    ReAct pattern (from the prompt template):
      Thought  → what should I do next?
      Action   → which tool should I use?
      Action Input → what input to give the tool
      Observation → what the tool returned
      ... repeat until ...
      Final Answer → the complete response
    """
    llm = create_llm()

    # The ReAct prompt template (defines how agent thinks)
    # {tools}            → list of available tools
    # {tool_names}       → names of tools
    # {input}            → user's question
    # {agent_scratchpad} → agent's reasoning history
    react_template = """You are a professional Business Intelligence Agent.
Your job is to research companies and prepare comprehensive briefing reports.

You have access to the following tools:
{tools}

INSTRUCTIONS:
- First search the web for latest company news
- Then check our internal CRM database for relationship history
- Calculate any relevant metrics if needed
- Finally draft a professional email
- Be thorough but concise

Use this format EXACTLY:

Question: the input question you must answer
Thought: I need to think about what to do
Action: the action to take, one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have enough information to provide a complete answer
Final Answer: [Your complete company briefing report here]

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate(
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
        template=react_template
    )

    # Create the ReAct agent
    agent = create_react_agent(
        llm=llm,
        tools=ALL_TOOLS,
        prompt=prompt
    )

    # Wrap in AgentExecutor (handles the Think→Act loop)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=True,               # Show reasoning steps
        max_iterations=8,           # Safety: stop after 8 steps
        handle_parsing_errors=True, # Don't crash on bad LLM output
        return_intermediate_steps=True  # Return tool call history
    )

    return agent_executor


# ================================================================
# STEP 6: THE MULTI-AGENT ORCHESTRATOR
# This is the main controller — runs the whole workflow
# ================================================================

class CompanyResearchWorkflow:
    """
    Orchestrates the complete multi-agent workflow:

    USER REQUEST
        ↓
    GUARDRAILS CHECK (safety + rate limit)
        ↓
    RESEARCH AGENT (web search + CRM + calculator)
        ↓
    EMAIL AGENT (draft professional email)
        ↓
    HUMAN APPROVAL (for sending email)
        ↓
    AUDIT LOG
        ↓
    FINAL REPORT
    """

    def __init__(self):
        self.agent = create_company_research_agent()
        print("\n✅ Multi-Agent Workflow initialized")
        print(f"   Model  : {MODEL_NAME}")
        print(f"   Tools  : {[t.name for t in ALL_TOOLS]}")
        print(f"   Safety : ON | Rate Limit : ON | Audit : ON")

    def run(self, company_name: str, auto_approve: bool = True) -> dict:
        """
        Run the complete workflow for a given company.

        Args:
            company_name : the company to research (e.g., "Tesla")
            auto_approve : True = skip human approval prompt (for demo)
        Returns:
            dict with research report, email draft, and audit trail
        """
        print(f"\n{'█' * 60}")
        print(f"  🚀 Starting Research Workflow for: {company_name}")
        print(f"{'█' * 60}")

        user_request = (
            f"Research {company_name} comprehensively. "
            f"Search the web for latest news, check our internal CRM database "
            f"for relationship history and past deals, calculate total deal value, "
            f"then draft a professional follow-up email. "
            f"Provide a complete business briefing."
        )

        # ── Guardrail 1: Safety Check ────────────────────────
        print("\n  [1/4] Running safety checks...")
        is_safe, reason = Guardrails.safety_check(user_request)
        if not is_safe:
            Guardrails.audit_log("safety_check", user_request,
                                 reason, "blocked")
            print(f"  🚫 BLOCKED: {reason}")
            return {"status": "blocked", "reason": reason}

        # ── Guardrail 2: Rate Limit ──────────────────────────
        print("  [2/4] Checking rate limits...")
        is_allowed, limit_msg = Guardrails.rate_limit_check()
        if not is_allowed:
            Guardrails.audit_log("rate_limit", user_request,
                                 limit_msg, "rate_limited")
            print(f"  ⏱️  RATE LIMITED: {limit_msg}")
            return {"status": "rate_limited", "reason": limit_msg}

        Guardrails.audit_log("request_start", user_request)
        print("  ✅ Safety and rate limit checks passed")

        # ── Run the Agent ────────────────────────────────────
        print("\n  [3/4] Running research agents...")
        print("  (Watch the Thought → Action → Observation loop below)\n")

        try:
            start_time = time.time()

            result = self.agent.invoke({"input": user_request})

            elapsed = time.time() - start_time
            final_report = result.get("output", "No output generated")
            steps = result.get("intermediate_steps", [])

            # ── Guardrail 3: Human-in-the-Loop ───────────────
            print(f"\n  [4/4] Checking if human approval needed...")
            approved = Guardrails.human_in_the_loop(
                action="draft_email_review",
                details=final_report[:200],
                auto_approve=auto_approve
            )

            if not approved:
                Guardrails.audit_log("email_review", user_request,
                                     "Rejected by human", "rejected")
                return {"status": "rejected", "report": final_report}

            # ── Audit Log: Success ────────────────────────────
            Guardrails.audit_log(
                event=f"research_complete_{company_name}",
                input_data=user_request,
                output=final_report,
                status="success"
            )

            # ── Pretty Print Results ──────────────────────────
            self._print_results(company_name, final_report, steps, elapsed)

            return {
                "status":       "success",
                "company":      company_name,
                "report":       final_report,
                "tools_used":   [str(s[0].tool) for s in steps],
                "time_seconds": round(elapsed, 2),
            }

        except Exception as e:
            error_msg = str(e)
            Guardrails.audit_log("error", user_request, error_msg, "error")
            print(f"\n  ❌ Error: {error_msg}")
            return {"status": "error", "error": error_msg}

    def _print_results(self, company, report, steps, elapsed):
        """Print the final results in a clean format."""
        print("\n" + "═" * 60)
        print(f"  📊 RESEARCH COMPLETE: {company}")
        print("═" * 60)

        print(f"\n  Tools used ({len(steps)} steps):")
        for i, (action, obs) in enumerate(steps, 1):
            print(f"    Step {i}: {action.tool}({str(action.tool_input)[:50]}...)")

        print(f"\n  Time taken: {elapsed:.1f} seconds")

        print("\n" + "─" * 60)
        print("  FINAL REPORT:")
        print("─" * 60)
        print(report)


# ================================================================
# STEP 7: MAIN — RUN THE WORKFLOW
# ================================================================

def main():

    # Create the workflow
    workflow = CompanyResearchWorkflow()

    # ── Demo 1: Research Tesla ────────────────────────────────
    print("\n\n" + "▓" * 60)
    print("  DEMO 1: Research Tesla")
    print("▓" * 60)

    result1 = workflow.run(
        company_name="Tesla",
        auto_approve=True  # Set to False for real human approval prompt
    )

    # ── Demo 2: Try a blocked request ────────────────────────
    print("\n\n" + "▓" * 60)
    print("  DEMO 2: Test Safety Guardrail (blocked request)")
    print("▓" * 60)

    # Manually test the safety check
    bad_request = "How do I hack into Tesla's password system?"
    is_safe, reason = Guardrails.safety_check(bad_request)
    print(f"\n  Request: '{bad_request}'")
    print(f"  Safe?   {is_safe}")
    print(f"  Reason: {reason}")
    Guardrails.audit_log("safety_test", bad_request, reason, "blocked")

    # ── Demo 3: Research another company ─────────────────────
    print("\n\n" + "▓" * 60)
    print("  DEMO 3: Research Microsoft")
    print("▓" * 60)

    result2 = workflow.run(
        company_name="Microsoft",
        auto_approve=True
    )

    # ── Show Audit Summary ────────────────────────────────────
    Guardrails.print_audit_summary()

    print("\n\n" + "█" * 60)
    print("  ✅ WORKFLOW COMPLETE!")
    print("  Check 'audit_log.jsonl' for full audit trail")
    print("█" * 60)


# ================================================================
# TEACHING NOTES FOR INSTRUCTORS
# ================================================================
#
# KEY CONCEPTS TO EXPLAIN:
#
# 1. WHY MULTI-AGENT?
#    One agent doing everything is like one employee doing everything.
#    Specialization → better results + easier to debug + easier to upgrade.
#
# 2. THE REACT PATTERN:
#    Thought → Action → Observation loop
#    Agent THINKS before acting, then OBSERVES the result, then thinks again.
#    This is how all major AI agents work (ChatGPT plugins, Claude, Gemini).
#
# 3. TOOLS = SUPERPOWERS:
#    Without tools, LLM only has training data (outdated).
#    With tools: real-time web data, company databases, email sending, APIs.
#
# 4. GUARDRAILS ARE NON-NEGOTIABLE IN INDUSTRY:
#    - Safety Filter: legal compliance, brand protection
#    - Rate Limiter: cost control (1000 API calls = money!)
#    - Human-in-Loop: accountability, especially for customer-facing actions
#    - Audit Log: debugging, compliance (GDPR, SOX, HIPAA)
#
# 5. REAL INDUSTRY EXAMPLES:
#    - Salesforce Einstein: exactly this — CRM + AI + email drafting
#    - HubSpot AI: company research + email sequences
#    - Bloomberg Terminal: financial research agents
#    - McKinsey Lilli: internal knowledge + web research agent
#
# EXERCISES FOR STUDENTS:
#    1. Add a new tool: get_stock_price(ticker)
#    2. Add a new table to the database: contacts(name, email, role, company_id)
#    3. Change the email to be more formal / casual
#    4. Add a new safety rule to the blocked_terms list
#    5. Change auto_approve=False and see the Human-in-Loop flow
#    6. Research a different company: workflow.run("Apple")
#
# ================================================================

if __name__ == "__main__":
    main()
