# Boskalis GeoChat Assistant v1

A next-generation, agentic chatbot for geotechnical data analysis and compliance reporting. Built for field engineers, supervisors, and project managers, it combines advanced LLMs with domain-specific automation to deliver actionable insights from raw settlement plate data.

v1: ollama context based chatbot
v2: ollama fine-tuned chatbot (in progress)

---

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gavkujo/NL2Func.git
   cd custom
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app:**
   ```bash
   streamlit run app.py
   ```
4. **Upload your data** and start chatting in natural language!

---

## Functional Features

This MVP chatbot is designed for geotechnical data analysis with three distinct operational modes:

1. **Base Mode** – Utilizes Google Gemma 2B, optimized for fast, general-purpose queries.
2. **Think Mode** (`@think`) – Runs DeepSeek R1 1.5B, specialized for complex reasoning tasks (best suited for in-depth analyses rather than casual conversation).
3. **Deep Mode** (`@deep`) – Employs DeepSeek R1 7B Q4 for high-accuracy reasoning and broader analysis capability, trading speed for depth.

**Additional Control Modes**

* **`@recap` Mode** – Brings prior conversation history into the context on demand, enabling informed follow-up queries while conserving memory.
* **Progressive Summarization of Memory** – Unique context management method where older exchanges are intelligently summarized, maintaining only a few context turns at full detail. This keeps token usage low and is patent processing.

**Agentic Functions** – The system dynamically detects prompts and executes one of four specialized tasks:

1. Asaoka method PDF report generation.
2. Settlement plate time-series plotting.
3. Multi-plate compliance and trend analysis.
4. Deep, single-plate geotechnical diagnostics.

---

## Geotechnical Usefulness

The MVP addresses key bottlenecks in large-scale ground improvement projects by:

1. **Parsing standard JSON settlement plate datasets** directly from site monitoring systems.
2. **Applying geotechnical compliance rules** (Asaoka DOC > 90%, ≤ 4 mm/7 days settlement rate, min. ground level 16.9 mCD).
3. **Delivering immediate, structured insights** that convert hundreds of readings into prioritized compliance reports.
4. **Preserving site context through conversational memory**, enabling longitudinal trend comparison without reloading data.
5. **Deploying in the field via a Streamlit UI**, allowing engineers, supervisors, and managers to query in natural language and receive technically correct, domain-aware responses.

---

## Non-Functional Performance Metrics

| Mode  | Steady-State Response Time | Startup Response Time* | Accuracy (BERT Score est.) |
| ----- | -------------------------- | ----------------------- | -------------------------- |
| Base  | ~10 s                      | 30 s                   | 80–85%                     |
| Think | ~15 s                      | 30–45 s                | 80–85%                     |
| Deep  | ~20–25 s                   | 60–90 s                | 80–85%                     |

*Startup time = first prompt sent to the model after load.

---

## Architecture & Tech Stack

- **Python 3.11+**
- **Streamlit** for the web UI
- **PyTorch** for model inference
- **Transformers** and **SentencePiece** for LLM and tokenization
- **Custom MiniTransformer** for function classification
- **Plotly**, **ReportLab** for plotting and PDF generation
- **Ollama** (optional) for local LLM serving

---

## Project Structure

```
custom/
├── app.py                # Streamlit UI
├── dispatcher.py         # Agentic function dispatcher
├── functions.py          # Geotechnical function handlers
├── infer.py              # Model inference logic
├── llm_main.py           # LLM routing and summarization
├── main.py               # Classifier and rule-based logic
├── requirements.txt      # Python dependencies
├── data/
│   ├── data_modeller.py
│   ├── full_dataset.json
│   └── parser_test.py    # Plate ID/date parsing
├── helpers/              # Data helpers and report generators (function implementations)
├── models/               # MiniTransformer model (for classification)
├── saved/                # Trained model checkpoints (for classification)
├── static/               # Generated PDFs, plots, and assets
├── t5-small/             # LLM weights/configs (summarizer)
├── tokenizer/            # SentencePiece model 
```

---

## Testing

- Run `python main.py` for command-line testing of the classifier and agentic pipeline.
- Unit tests for parsing and function mapping are in `data/parser_test.py`.

---

## Future Work

1. **Model Fine-Tuning** – Train on historical geotechnical datasets to improve accuracy, reduce ambiguity, and align outputs with project-specific language. Costs limited to temporary GPU instance rental during training.
2. **Hosting More Powerful Models** – Deploy optimized larger models on cloud instances for higher precision and reduced reasoning latency. This is easier to implement than fine-tuning but requires ongoing hosting costs.


---

## Contact

For questions, bug reports, or collaboration, contact [garv.sachdev@boskalis.com].
