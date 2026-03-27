"""
RunnableParallel for Multi-Output Clinical Analysis
=====================================================
Demonstrates using RunnableParallel to run multiple analysis chains
simultaneously on the same clinical note. Instead of running urgency
classification, ICD-10 coding, and summarization sequentially (3x latency),
we run them in parallel (~1x latency).

This is one of the biggest practical advantages of LCEL for healthcare AI:
clinical workflows often need multiple different analyses of the same input.

Requires: OPENAI_API_KEY environment variable, langchain, langchain-openai
"""

import json
import time
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


# --- Output Schemas ---

class UrgencyAssessment(BaseModel):
    """Urgency classification output."""
    urgency_level: int = Field(ge=1, le=5)
    urgency_label: str
    reasoning: str
    disposition: str


class ICD10Coding(BaseModel):
    """ICD-10 coding output."""
    primary_code: str
    primary_description: str
    secondary_codes: list[dict[str, str]] = Field(default_factory=list)


class ClinicalSummary(BaseModel):
    """Structured clinical summary."""
    one_liner: str = Field(description="Single-sentence clinical summary")
    key_findings: list[str]
    red_flags: list[str]


# --- Individual Chain Definitions ---

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Chain A: Urgency Classification
urgency_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a triage urgency classifier. Classify the urgency of the clinical note.\n"
        "Scale: 1=ROUTINE, 2=LOW, 3=MODERATE, 4=HIGH, 5=EMERGENT.\n"
        "Return JSON: urgency_level (int), urgency_label (str), reasoning (str), "
        "disposition (str: 'immediate_ed'|'urgent_care'|'same_day'|'scheduled'|'telehealth').\n"
        "{format_instructions}",
    ),
    ("human", "Classify urgency:\n\n{note_text}"),
])
urgency_parser = JsonOutputParser(pydantic_object=UrgencyAssessment)
urgency_chain = (
    {"note_text": lambda x: x["note_text"], "format_instructions": lambda _: urgency_parser.get_format_instructions()}
    | urgency_prompt
    | model
    | urgency_parser
)

# Chain B: ICD-10 Coding
icd10_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a medical coding specialist. Assign ICD-10-CM codes to the clinical note.\n"
        "Provide the most specific primary code and up to 3 secondary codes.\n"
        "Return JSON: primary_code (str), primary_description (str), "
        "secondary_codes (list of {{code, description}}).\n"
        "{format_instructions}",
    ),
    ("human", "Assign ICD-10 codes:\n\n{note_text}"),
])
icd10_parser = JsonOutputParser(pydantic_object=ICD10Coding)
icd10_chain = (
    {"note_text": lambda x: x["note_text"], "format_instructions": lambda _: icd10_parser.get_format_instructions()}
    | icd10_prompt
    | model
    | icd10_parser
)

# Chain C: Clinical Summary
summary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a clinical documentation specialist. Produce a structured summary.\n"
        "Return JSON: one_liner (single-sentence summary), key_findings (list of strings), "
        "red_flags (list of strings, empty if none).\n"
        "{format_instructions}",
    ),
    ("human", "Summarize:\n\n{note_text}"),
])
summary_parser = JsonOutputParser(pydantic_object=ClinicalSummary)
summary_chain = (
    {"note_text": lambda x: x["note_text"], "format_instructions": lambda _: summary_parser.get_format_instructions()}
    | summary_prompt
    | model
    | summary_parser
)


# --- Parallel Composition ---

parallel_analysis = RunnableParallel(
    urgency=urgency_chain,
    icd10=icd10_chain,
    summary=summary_chain,
)


def merge_results(results: dict[str, Any]) -> dict[str, Any]:
    """Merge parallel results into a single unified output."""
    return {
        "urgency": results.get("urgency", {}),
        "coding": results.get("icd10", {}),
        "summary": results.get("summary", {}),
    }


# Full pipeline: parallel analysis then merge
full_pipeline = parallel_analysis | RunnableLambda(merge_results)


# --- Sequential Comparison (for timing) ---

def run_sequential(note_text: str) -> tuple[dict, float]:
    """Run all three chains sequentially and measure time."""
    start = time.time()
    input_dict = {"note_text": note_text}
    urgency_result = urgency_chain.invoke(input_dict)
    icd10_result = icd10_chain.invoke(input_dict)
    summary_result = summary_chain.invoke(input_dict)
    elapsed = time.time() - start
    return {
        "urgency": urgency_result,
        "coding": icd10_result,
        "summary": summary_result,
    }, elapsed


def run_parallel(note_text: str) -> tuple[dict, float]:
    """Run all three chains in parallel and measure time."""
    start = time.time()
    result = full_pipeline.invoke({"note_text": note_text})
    elapsed = time.time() - start
    return result, elapsed


# --- Test Data ---

TEST_NOTE: str = (
    "78-year-old female with history of atrial fibrillation on apixaban, CHF (EF 30%), "
    "CKD stage 3b, and type 2 diabetes presents with 3 days of worsening dyspnea, "
    "productive cough with yellow-green sputum, and fever to 102.1F. Unable to sleep flat "
    "for 2 nights. Increasing leg swelling. Vitals: BP 98/62, HR 108 irregular, RR 26, "
    "SpO2 88% on room air (improved to 93% on 4L NC), Temp 102.1F. Exam: ill-appearing, "
    "JVD, bilateral crackles to mid-lungs, irregular tachycardic rhythm, 3+ bilateral pitting "
    "edema. Labs: WBC 16.8, Hgb 10.2, Cr 2.4 (baseline 1.8), proBNP 8400, procalcitonin 2.8, "
    "lactate 2.6. CXR: bilateral pulmonary infiltrates with bilateral pleural effusions. "
    "Blood cultures drawn. Started on ceftriaxone and azithromycin."
)


def main() -> None:
    """Compare sequential vs parallel execution and display results."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich import box

        console = Console()
        use_rich = True
    except ImportError:
        use_rich = False

    if use_rich:
        console.print(
            Panel(
                TEST_NOTE[:250] + "...",
                title="[bold]Test Clinical Note[/bold]",
                border_style="cyan",
            )
        )

    # Run sequential
    if use_rich:
        console.print("\n[bold yellow]Running Sequential (3 API calls in series)...[/bold yellow]")
    seq_result, seq_time = run_sequential(TEST_NOTE)

    # Run parallel
    if use_rich:
        console.print("[bold green]Running Parallel (3 API calls simultaneously)...[/bold green]")
    par_result, par_time = run_parallel(TEST_NOTE)

    if use_rich:
        # Timing comparison
        speedup = seq_time / par_time if par_time > 0 else 0
        console.print(f"\n[bold]Performance Comparison:[/bold]")
        console.print(f"  Sequential: {seq_time:.2f}s")
        console.print(f"  Parallel:   {par_time:.2f}s")
        console.print(f"  Speedup:    [green]{speedup:.1f}x[/green]")

        # Display parallel results
        result = par_result

        # Urgency
        urgency = result.get("urgency", {})
        u_level = urgency.get("urgency_level", "?")
        u_color = {1: "green", 2: "green", 3: "yellow", 4: "red", 5: "bold red"}.get(u_level, "white")
        console.print(f"\n[bold]Urgency:[/bold] [{u_color}]{u_level} - {urgency.get('urgency_label', '?')}[/{u_color}]")
        console.print(f"  Reasoning: {urgency.get('reasoning', 'N/A')}")
        console.print(f"  Disposition: {urgency.get('disposition', 'N/A')}")

        # ICD-10
        coding = result.get("coding", {})
        console.print(f"\n[bold]ICD-10 Coding:[/bold]")
        console.print(f"  Primary: {coding.get('primary_code', '?')} - {coding.get('primary_description', '?')}")
        for sec in coding.get("secondary_codes", []):
            console.print(f"  Secondary: {sec.get('code', '?')} - {sec.get('description', '?')}")

        # Summary
        summary = result.get("summary", {})
        console.print(f"\n[bold]Clinical Summary:[/bold]")
        console.print(f"  {summary.get('one_liner', 'N/A')}")
        console.print(f"  Key Findings: {', '.join(summary.get('key_findings', []))}")
        red_flags = summary.get("red_flags", [])
        if red_flags:
            console.print(f"  [red]Red Flags: {', '.join(red_flags)}[/red]")
    else:
        print(f"\nSequential: {seq_time:.2f}s | Parallel: {par_time:.2f}s")
        print(f"Urgency: {par_result.get('urgency', {}).get('urgency_level', '?')}")
        print(f"ICD-10: {par_result.get('coding', {}).get('primary_code', '?')}")


if __name__ == "__main__":
    main()


# --- Sample Output ---
#
# Test Clinical Note:
# 78-year-old female with history of atrial fibrillation on apixaban, CHF (EF 30%)...
#
# Running Sequential (3 API calls in series)...
# Running Parallel (3 API calls simultaneously)...
#
# Performance Comparison:
#   Sequential: 4.82s
#   Parallel:   1.94s
#   Speedup:    2.5x
#
# Urgency: 5 - EMERGENT
#   Reasoning: Elderly patient with multiple comorbidities presenting with likely pneumonia
#     superimposed on decompensated CHF. Hemodynamic instability (hypotension), hypoxia,
#     elevated lactate, and sepsis markers indicate high mortality risk.
#   Disposition: immediate_ed
#
# ICD-10 Coding:
#   Primary: J18.9 - Pneumonia, unspecified organism
#   Secondary: I50.23 - Acute on chronic systolic heart failure
#   Secondary: N18.3b - Chronic kidney disease, stage 3b
#   Secondary: A41.9 - Sepsis, unspecified organism
#
# Clinical Summary:
#   78F with CHF, AFib, CKD presenting with pneumonia-triggered acute decompensated
#   heart failure and early sepsis.
#   Key Findings: Bilateral infiltrates, elevated procalcitonin, proBNP 8400, lactate 2.6
#   Red Flags: Hypotension, hypoxia, elevated lactate, AKI on CKD
