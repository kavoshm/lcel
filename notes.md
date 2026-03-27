# LangChain Expression Language (LCEL) — Study Notes

## Core Concept: Runnables

Everything in LCEL is a **Runnable** — an object with `.invoke()`, `.stream()`, `.batch()`,
and async variants. The pipe operator chains Runnables together.

```python
# Each of these is a Runnable:
prompt = ChatPromptTemplate.from_messages(...)   # Runnable
model = ChatOpenAI(model="gpt-4o-mini")          # Runnable
parser = StrOutputParser()                        # Runnable

# Chain them with pipe:
chain = prompt | model | parser                   # Also a Runnable!
```

The beauty: the composed chain is itself a Runnable, so it has the same interface (.invoke,
.stream, .batch) and can be composed further.

---

## RunnablePassthrough

Passes the input through while optionally adding computed fields via `.assign()`.

```python
from langchain_core.runnables import RunnablePassthrough

# Add a "word_count" field to the input without changing existing fields
chain = RunnablePassthrough.assign(
    word_count=lambda x: len(x["note_text"].split())
) | prompt | model | parser
```

**Healthcare use case:** Add metadata (note length, timestamp, source system) to the input
before sending to the LLM.

---

## RunnableParallel

Runs multiple chains simultaneously on the same input and returns a dict of results.

```python
from langchain_core.runnables import RunnableParallel

parallel_chain = RunnableParallel(
    urgency=urgency_chain,
    icd10=icd10_chain,
    summary=summary_chain,
)

# All three chains run in parallel on the same input
results = parallel_chain.invoke({"note_text": "..."})
# results = {"urgency": ..., "icd10": ..., "summary": ...}
```

**Healthcare use case:** Classify urgency, extract ICD-10 codes, and generate a summary
all at once — reducing total latency from 3x sequential to ~1x parallel.

---

## RunnableLambda

Wraps any Python function as a Runnable:

```python
from langchain_core.runnables import RunnableLambda

def deidentify(input_dict: dict) -> dict:
    """Remove PHI from clinical note before LLM processing."""
    note = input_dict["note_text"]
    # In production: use a proper de-identification library
    cleaned = note.replace("John Smith", "[PATIENT]")
    return {**input_dict, "note_text": cleaned}

chain = RunnableLambda(deidentify) | prompt | model | parser
```

**Healthcare use case:** PHI de-identification, input validation, or custom preprocessing
before the LLM call.

---

## RunnableBranch

Conditional routing — choose which chain to execute based on the input:

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: x["specialty"] == "cardiology", cardiology_chain),
    (lambda x: x["specialty"] == "mental_health", psych_chain),
    general_chain,  # Default branch
)
```

**Healthcare use case:** Route clinical notes to specialty-specific analysis chains.

---

## Streaming

LCEL chains support token-by-token streaming natively:

```python
for chunk in chain.stream({"note_text": "..."}):
    print(chunk, end="", flush=True)
```

**Healthcare use case:** Real-time display of clinical analysis as the LLM generates it.
Users see tokens appearing live rather than waiting for the entire response.

### Streaming with Intermediate Steps

You can stream events from all steps in the chain:

```python
async for event in chain.astream_events({"note_text": "..."}, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
```

---

## Batch Processing

Process multiple inputs in parallel:

```python
notes = [
    {"note_text": "Note 1..."},
    {"note_text": "Note 2..."},
    {"note_text": "Note 3..."},
]

results = chain.batch(notes, config={"max_concurrency": 5})
```

`max_concurrency` controls parallelism — important for API rate limits.

**Healthcare use case:** Batch classify 100 clinical notes from a daily ingest. Set
concurrency based on your OpenAI rate limits.

---

## Composition Patterns

### Sequential Chain
```python
chain = step1 | step2 | step3
```

### Parallel then Merge
```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

def merge_results(results: dict) -> str:
    return f"Urgency: {results['urgency']}, ICD: {results['icd10']}"

chain = (
    RunnableParallel(urgency=urgency_chain, icd10=icd10_chain)
    | RunnableLambda(merge_results)
)
```

### Passthrough + Parallel
```python
chain = (
    RunnablePassthrough.assign(word_count=lambda x: len(x["note"].split()))
    | RunnableParallel(classification=classify_chain, summary=summarize_chain)
)
```

---

## LCEL vs Legacy Chains

| Feature | Legacy (LLMChain) | LCEL |
|---------|-------------------|------|
| Streaming | Manual | Built-in |
| Batch | Manual | Built-in |
| Async | Manual | Built-in |
| Composability | Nested classes | Pipe operator |
| Tracing | Requires setup | Automatic with LangSmith |
| Type hints | Weak | Strong |
| Learning curve | Moderate | Low (if you know Python) |

**Verdict:** Use LCEL for all new development. Legacy chains are maintained but not the future.

---

## Key Insights for Healthcare AI

1. **RunnableParallel is a performance multiplier.** Clinical analysis often involves multiple
   independent tasks (urgency, coding, summarization). Running them in parallel cuts latency
   by 60-70%.

2. **Streaming improves UX dramatically.** Clinicians do not want to wait 5 seconds for a
   blank screen to populate. Streaming shows results immediately.

3. **RunnableLambda bridges LLM and traditional code.** PHI de-identification, input validation,
   and output post-processing are traditional code — RunnableLambda lets you include them in
   the LCEL chain seamlessly.

4. **Batch + max_concurrency = controlled scale.** Processing 500 clinical notes? Batch with
   concurrency=10 processes them in about 50 sequential API calls worth of time, while
   respecting rate limits.

5. **Every LCEL step is automatically traced.** This is critical for healthcare audit
   requirements — you can see exactly what each step received and produced.
