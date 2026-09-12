# Tomato Leaf Clinic

Photograph a tomato leaf, get a disease name and a treatment plan that comes from a
written guide rather than from a language model's memory.

**Live demo:** _paste your Render URL here_
**Notebook that trained the model:** _paste your Colab link here_

## What it does

| Step | What runs | Why |
|---|---|---|
| 1 | CNN classifier fine-tuned on PlantVillage (2.26 M params) | Names the disease from pixels. An LLM cannot do this reliably. |
| 2 | FAISS semantic search over the treatment guide | Finds the right section by meaning, not keywords. |
| 3 | LangChain agent (Gemini 2.0 Flash) | Decides which of the two tools to call, then writes the answer. |

The agent is not hardcoded. If you send a question with no image it skips the CNN
entirely and only searches the guide. If you send an image it calls the CNN first
and feeds the predicted disease into the search.

Classes: `Tomato___Bacterial_spot`, `Tomato___Early_blight`, `Tomato___Late_blight`, `Tomato___healthy`.
Model input: 128×128 RGB. Validation accuracy: _fill in from your training run_.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | Status and class list. `503` if the model failed to load. |
| POST | `/predict` | `image` file | CNN prediction only, no LLM call |
| POST | `/ask` | `question` text, optional `image` file | Full answer plus the tools the agent used |

Interactive docs at `/docs`.

```bash
curl -X POST https://YOUR-SERVICE.onrender.com/predict -F "image=@leaf.jpg"
```

## Run it locally

`plant_disease_model.keras` and `class_names.json` are already committed here —
both came out of the training notebook.

```bash
cp .env.example .env          # then paste your Google API key into it
pip install -r requirements.txt
uvicorn main:app --reload --port 7860
```

Open http://localhost:7860

## Run it in Docker

```bash
docker build -t leaf-clinic .
docker run -p 7860:7860 -e GOOGLE_API_KEY=your_key leaf-clinic
```

## Deploy it

`render.yaml` is a Render Blueprint — push this repo, point Render at it, supply
`GOOGLE_API_KEY`, done. Step by step in **[DEPLOY.md](DEPLOY.md)**.

One number decides everything about hosting this: **the service idles at ~775 MB
of RAM.** TensorFlow is ~500 MB before the model is even loaded. Render's Free and
Starter instances cap at 512 MB and will be OOM-killed during startup, so the
Blueprint asks for `standard` (2 GB). DEPLOY.md covers the cheaper routes.

## Limits worth knowing

- Trained on four tomato classes only. Show it a potato leaf and it will confidently
  return one of the four — the model has no way to say "I don't know".
- PlantVillage images are single leaves on a plain background in good light. Accuracy
  on a real field photo with soil, shadows and multiple leaves is lower.
- Treatment advice is only as current as `plant_disease_guide.md`. Edit that file and
  the advice changes on the next restart, with no retraining.
- Startup takes 30–60 seconds: loading TensorFlow, then embedding the guide through
  Google's API. The service is not ready until `/health` returns 200.
