"""
LCEL Basics — Chain Composition for Clinical Analysis
======================================================
Demonstrates the fundamentals of LCEL (LangChain Expression Language):
- Building chains with the pipe operator
- Using ChatPromptTemplate and output parsers
- RunnablePassthrough for adding metadata
- RunnableLambda for custom processing steps
- Comparing LCEL to legacy chain patterns

Requires: OPENAI_API_KEY environment variable, langchain, langchain-openai
"""

from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


# --- Output Schema ---

class ClinicalExtraction(BaseModel):
    """Structured extraction from a clinical note."""
    chief_complaint: str = Field(description="Primary reason for visit")
    symptoms: list[str] = Field(description="List of symptoms mentioned")
    vital_sign_abnormalities: list[str] = Field(description="Any abnormal vitals noted")
    urgency_assessment: str = Field(description="Brief urgency assessment")
    urgency_level: int = Field(ge=1, le=5, description="Urgency 1-5")


# --- Chain Components ---

extraction_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a clinical data extraction system. Extract structured information "
        "from the clinical note. Return JSON matching the requested schema.\n"
        "{format_instructions}",
    ),
    (
        "human",
        "Extract clinical information from this note "
        "(word count: {word_count} words):\n\n{note_text}",
    ),
])

summary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a clinical summarization specialist. Produce a 2-3 sentence "
        "summary of the clinical note suitable for a handoff report.",
    ),
    ("human", "Summarize this clinical note:\n\n{note_text}"),
])

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
json_parser = JsonOutputParser(pydantic_object=ClinicalExtraction)
str_parser = StrOutputParser()


# --- Custom Processing Functions ---

def add_metadata(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Add computed metadata fields to the input (preprocessing step)."""
    note = input_dict.get("note_text", "")
    return {
        **input_dict,
        "word_count": len(note.split()),
        "char_count": len(note),
        "has_vitals": any(
            term in note.lower()
            for term in ["bp ", "hr ", "rr ", "spo2", "temp ", "blood pressure"]
        ),
    }


def validate_output(result: dict) -> dict:
    """Post-process and validate the extraction output."""
    # Ensure urgency_level is within bounds
    urgency = result.get("urgency_level", 3)
    result["urgency_level"] = max(1, min(5, urgency))

    # Add urgency label
    labels = {1: "ROUTINE", 2: "LOW", 3: "MODERATE", 4: "HIGH", 5: "EMERGENT"}
    result["urgency_label"] = labels.get(result["urgency_level"], "UNKNOWN")

    return result


# --- LCEL Chain Composition ---

# Chain 1: Basic extraction chain
extraction_chain = (
    RunnableLambda(add_metadata)                           # Step 1: Add metadata
    | RunnablePassthrough.assign(                          # Step 2: Add format instructions
        format_instructions=lambda _: json_parser.get_format_instructions()
    )
    | extraction_prompt                                     # Step 3: Format prompt
    | model                                                 # Step 4: LLM call
    | json_parser                                           # Step 5: Parse JSON output
    | RunnableLambda(validate_output)                       # Step 6: Validate and enhance
)

# Chain 2: Simple summarization chain
summary_chain = summary_prompt | model | str_parser


# --- Test Data ---

TEST_NOTES: list[dict[str, str]] = [
    {
        "label": "Emergency Presentation",
        "note_text": (
            "67-year-old male presenting with acute onset severe substernal chest pain "
            "radiating to jaw, associated diaphoresis and dyspnea. PMH: DM2, HTN, "
            "hyperlipidemia. BP 92/58, HR 118, RR 24, SpO2 90% on room air. ECG shows "
            "ST elevation V1-V4. Troponin 4.2. Aspirin and heparin administered. "
            "Cardiology activated for emergent PCI."
        ),
    },
    {
        "label": "Routine Visit",
        "note_text": (
            "44-year-old female presents for annual wellness exam. No active complaints. "
            "Regular exercise 4x/week, balanced diet. Non-smoker, occasional alcohol. "
            "Vitals: BP 118/72, HR 68, BMI 23.4. Physical exam unremarkable. Labs ordered: "
            "CBC, CMP, lipid panel, TSH. Cervical cancer screening current. Flu vaccine "
            "administered. Return in 1 year."
        ),
    },
    {
        "label": "Moderate Acuity",
        "note_text": (
            "52-year-old male with known gout presents with acute swelling, redness, and "
            "severe pain in right first MTP joint, onset 12 hours ago. Unable to bear weight. "
            "Reports dietary indiscretion (red meat, beer) over the weekend. Last gout flare "
            "6 months ago. Currently on allopurinol 300mg daily. No fever. Vitals stable. "
            "Joint aspirate shows negatively birefringent needle-shaped crystals."
        ),
    },
]


def main() -> None:
    """Demonstrate LCEL chain composition on clinical notes."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
        use_rich = True
    except ImportError:
        use_rich = False

    for note_data in TEST_NOTES:
        input_dict = {"note_text": note_data["note_text"]}

        # Run extraction chain
        extraction_result = extraction_chain.invoke(input_dict)

        # Run summary chain
        summary_result = summary_chain.invoke(input_dict)

        if use_rich:
            console.print(f"\n[bold cyan]{'=' * 80}[/bold cyan]")
            console.print(f"[bold]{note_data['label']}[/bold]")
            console.print(
                Panel(note_data["note_text"][:200] + "...", title="Input", border_style="dim")
            )

            # Extraction results
            table = Table(title="Structured Extraction")
            table.add_column("Field", style="bold")
            table.add_column("Value")

            table.add_row("Chief Complaint", extraction_result.get("chief_complaint", "N/A"))
            table.add_row(
                "Symptoms", ", ".join(extraction_result.get("symptoms", []))
            )
            table.add_row(
                "Vital Abnormalities",
                ", ".join(extraction_result.get("vital_sign_abnormalities", ["None"])),
            )
            table.add_row("Urgency Assessment", extraction_result.get("urgency_assessment", "N/A"))

            urgency = extraction_result.get("urgency_level", 0)
            label = extraction_result.get("urgency_label", "?")
            color = {1: "green", 2: "green", 3: "yellow", 4: "red", 5: "bold red"}.get(urgency, "white")
            table.add_row("Urgency Level", f"[{color}]{urgency} ({label})[/{color}]")

            console.print(table)

            # Summary
            console.print(f"\n[bold]Summary:[/bold] {summary_result}")
        else:
            print(f"\n{note_data['label']}:")
            print(f"  Chief Complaint: {extraction_result.get('chief_complaint', 'N/A')}")
            print(f"  Urgency: {extraction_result.get('urgency_level', '?')}")
            print(f"  Summary: {summary_result}")


if __name__ == "__main__":
    main()


# --- Sample Output ---
#
# Emergency Presentation
# ╭─── Input ───╮
# │ 67-year-old male presenting with acute onset severe substernal chest pain...
# ╰─────────────╯
#
#        Structured Extraction
# ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Field                ┃ Value                                                ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │ Chief Complaint      │ Acute substernal chest pain with radiation to jaw    │
# │ Symptoms             │ chest pain, diaphoresis, dyspnea                     │
# │ Vital Abnormalities  │ Hypotension (92/58), Tachycardia (118), Hypoxia (90%)│
# │ Urgency Assessment   │ STEMI with hemodynamic instability, life-threatening │
# │ Urgency Level        │ 5 (EMERGENT)                                         │
# └──────────────────────┴──────────────────────────────────────────────────────┘
#
# Summary: 67-year-old male with STEMI (ST elevation V1-V4, troponin 4.2) presenting
#   with chest pain and hemodynamic instability. Emergent PCI activated.
#
# Routine Visit
# Urgency Level: 1 (ROUTINE)
# Summary: Healthy 44-year-old female presenting for annual wellness exam with no complaints,
#   normal vitals, and unremarkable physical exam. Routine labs ordered and flu vaccine given.
