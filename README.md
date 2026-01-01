# DATA3800-WattTommorrow-clean-
Semester project for DATA3800 - Data Science with Scripting. This is the cleaned up version, as the actual repository has alot of unused code.

Forecasting electricity prices and explaining them with a local GenAI assistant.

---

## Project Structure

```text
root
│
├── src/
│   ├── main.py
│   ├── main.ipynb
│   ├── static/
│   │   └── index.html
│   └── best_model/
│       ├── model.joblib
│       └── model_meta.json
│
├── data/
│   ├── ES/
│   │   └── predictions/
│   │       └── ES_predictions.csv
│   ├── energy_dataset.csv
│   └── weather_features.csv
│
├── README.md
└── requirements.txt

````

* **src/main.ipynb** – Full EDA, feature engineering, model training and plotting.
* **src/main.py** – FastAPI app exposing prediction + GenAI summarization endpoints.
* **src/static/index.html** – Simple UI to interact with predictions and summaries in the browser.
* **data/** – Raw and processed CSV files used for training and prediction.

---

## Libraries & Installation

The code is designed for python version 3.12.0

All required Python packages are listed in **requirements.txt**.

```bash
pip install -r requirements.txt
```

>  Recommended: use a virtual environment (`python -m venv venv && source venv/bin/activate` on Linux/macOS).

---

## How to Run

### 1. Run the Notebook (Data & Model)

You can execute `src/main.ipynb` from top to bottom to:

* Perform preprocessing and feature engineering
* Train the final XGBoost/ML model
* Save:

  * `best_model/model.joblib`
  * `best_model/model_meta.json`
  * Prepared prediction rows in `data/ES/predictions/ES_predictions.csv`

These prepared CSVs are what the FastAPI app will use for **read-only prediction lookup**.

---

### 2. Run the FastAPI + GenAI App

From the project root:

```bash
uvicorn src.main:app --reload
```

Then open:

```text
http://localhost:8000
```

The browser UI lets you:

1. Load a prediction by ID
2. Generate an ML prediction
3. Ask a GenAI model to **summarize and explain** the predicted price

---

## GenAI – Local LLM Summarization

The project includes a **local LLM** endpoint that generates short explanations for the predicted electricity price.

* Endpoint: `POST /summarize`
* Location: implemented in **src/main.py**
* Runs **entirely on CPU** – no external API calls, no keys.

The summarization model receives:

* `predicted_price` (float)
* `user_query` (string question about the price)

and returns a short, human-readable explanation.

---

### Supported Local Models (Tested)

We have tested three HuggingFace community models:

* **Microsoft Phi-3 Mini (4K Instruct)**
* **TinyLlama-1.1B-Chat** (light LLaMA-based model)
* **Qwen2-1.5B-Instruct**

All of them are:

* Fully **open source**
* Publicly available on HuggingFace
* **No HuggingFace token required** (for CPU usage via `transformers`)

---

### Switching Between LLMs

Inside `src/main.py`, each model has its own **model-loading block**.
Example (Phi-3):

```python
'''
DEVICE = "cpu"
LLM_NAME = "microsoft/phi-3-mini-4k-instruct"
tokenizer_llm = AutoTokenizer.from_pretrained(LLM_NAME)
model_llm = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    torch_dtype=torch.float32,
    device_map="cpu"
)
'''
```

Only **one model can be active at a time**.

**To switch models:**

1. **Uncomment** the block for the model you want to use
2. **Comment out** the other model blocks
3. Save `main.py`
4. Restart Uvicorn

---

### Summarization Endpoint Structure

Because each LLM has its own prompting style, `main.py` includes **multiple versions** of:

```python
@app.post("/summarize")
```

Each version is clearly labeled, such as:

* `# Phi-3 Summarization`
* `# TinyLlama Summarization`
* `# Qwen Summarization`

**To select one:**

* Uncomment the desired `@app.post("/summarize")` implementation
* Comment out the others

This prevents FastAPI route conflicts and ensures the correct **prompt template** is used for each model.

---

### Hardware Recommendations

Local models are memory-hungry. For stable performance:

* **8 CPU cores** (or more)
* **24 GB RAM** (or more, especially for Phi-3)
* Linux environment strongly recommended

Example observed with Phi-3 Mini during testing:

* CPU usage: ~76%
* Memory usage: ~39% of 32 GB (≈ 12.8 GB)

---

## Testing the `/summarize` Endpoint

With Uvicorn running:

```bash
uvicorn src.main:app --reload
```

### Using `curl`

```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
     -H "Content-Type: application/json" \
     -d '{"predicted_price": 92.5, "user_query": "Is this expensive for households?"}'
```

### Using the Browser UI

Open:

```text
http://localhost:8000
```

Then follow the guided steps on the page.

---

## End-to-End Usage Flow in the Browser

1. **First load the ID**
   Use the dropdown to select which prediction ID you want to inspect.

2. **Check which ID needs to be retrieved**
   The app fetches the corresponding row from the CSV and displays its contents.

3. **Add the prediction into the summarization box**
   The predicted price is copied/typed into the summarization form.

4. **Ask your question regarding the prediction**
   For example:
   *“Is this electricity price high or low for a typical home?”*
   The GenAI endpoint returns a short explanation based on that price.

---

## GenAI Model Comparison – Summary

| Model                                | Size / Requirements                                 | Performance During Test                      | Answer Quality & Behaviour                                          | Notes                                                     |
| ------------------------------------ | --------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------- |
| **Microsoft Phi-3 Mini 4K Instruct** | ~4B parameters; needs **8+ CPU** & **12–16 GB RAM** | CPU ~76%, RAM ~39% of 32 GB (~12.8 GB used)  | Very detailed, clearly states price is **high**, sometimes verbose  | Strong reasoning, but often repeats and overwrites points |
| **Qwen/Qwen2-1.5B-Instruct**         | ~3.09 GB; medium-size, easy to run, fully open      | Smooth and stable on a single machine        | Balanced, well-structured, moderate length, “normal” classification | Good default choice between quality and resource usage    |
| **TinyLlama-1.1B-Chat**              | Very small (1.1B), extremely lightweight            | Very fast inference, minimal hardware needed | Declares price **high**, but reasoning is more generic              | Great for speed; weaker precision and nuance              |

---

## Example Question Used for Evaluation

> **“Is this electricity price (69.83 EUR/MWh) high or low for a typical home?”**

**Observed behaviour:**

* **Phi-3 Mini**

  * Classifies it as **high**
  * Delivers long, detailed explanation
  * Sometimes repeats similar sentences

* **Qwen2-1.5B-Instruct**

  * Treats it as **normal** in context
  * Provides balanced and nuanced explanation
  * Emphasizes uncertainty and multiple factors (market, weather, etc.)

* **TinyLlama-1.1B-Chat**

  * Calls it **high**
  * Simple, shorter reasoning
  * Less accurate in terms of market realism

---