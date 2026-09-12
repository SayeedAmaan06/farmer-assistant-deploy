# Deploying the Tomato Leaf Clinic on Render with Docker

Follow this in order. Every step ends with something you can check, so you always
know whether to move on or to stop and fix.

---

## Before you start

1. A free Google AI Studio API key — https://aistudio.google.com/apikey
2. A GitHub account, with this repository pushed to it
3. A Render account — https://dashboard.render.com/register
4. Docker Desktop, if you want to test the container before you deploy it

`plant_disease_model.keras` and `class_names.json` are already committed, so there
is nothing to download from the training notebook.

---

## The one thing to know before you start clicking

This service needs about **775 MB of RAM at idle**, measured after startup with one
prediction served. The breakdown:

| What | RSS |
|---|---|
| `import tensorflow` | ~500 MB |
| loading the `.keras` model | ~45 MB |
| first prediction (XLA compile) | ~130 MB |
| FAISS + LangChain + FastAPI | ~100 MB |

Render's instance types:

| Plan | RAM | Will this app run? |
|---|---|---|
| Free | 512 MB | **No.** OOM-killed during startup. |
| Starter ($7/mo) | 512 MB | **No.** Same 512 MB as Free. |
| **Standard ($25/mo)** | **2 GB** | **Yes.** What `render.yaml` asks for. |

Paying $25 for a student project is a real decision, so make it deliberately rather
than by surprise at the end of a failed deploy. Your options:

- **Deploy on Standard.** It works, it stays awake, and you can suspend the service
  from the dashboard between demos so you are not billed for idle weeks.
- **Drop TensorFlow.** Convert the model to TFLite or ONNX and swap the runtime in
  `agent.py`. The interpreter is tens of megabytes instead of ~500, which brings the
  whole service under 512 MB and onto the free tier. This is a genuine code change,
  not a config flag — see "Getting onto the free tier" at the bottom.
- **Use Hugging Face Spaces instead.** Free Docker Spaces get 16 GB of RAM. Same
  Dockerfile, no code change. It is the pragmatic choice if the goal is a public
  link rather than Render specifically.

---

## Step 1 — Run it on your own machine

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # Windows: copy .env.example .env
```

Open `.env` and paste your key after `GOOGLE_API_KEY=`. Then:

```bash
uvicorn main:app --reload --port 7860
```

Startup takes 30–60 seconds. It loads TensorFlow, loads the CNN, then sends the
treatment guide to Google's embedding API to build the FAISS index.

**Check:** open http://localhost:7860/health — it should say `"model_loaded": true`.
If it does, open http://localhost:7860 and diagnose a leaf.

If you get a `503` instead, read the `error` field in that same JSON response. It
names the actual exception, which is nine times out of ten a missing key or a key
pasted with quotes around it.

---

## Step 2 — Put it in a container

```bash
docker build -t leaf-clinic .
```

The first build takes five to ten minutes, mostly downloading TensorFlow. Later
builds take seconds, because Docker caches the dependency layer — that is exactly
why `requirements.txt` is copied before the rest of the code.

```bash
docker run -p 7860:7860 -e GOOGLE_API_KEY=your_real_key leaf-clinic
```

**Check:** http://localhost:7860 works again, but this time nothing on your machine
is involved except Docker. No Python, no virtualenv, no "works on my laptop".

Read `-p 7860:7860` as: port 7860 on my machine connects to port 7860 inside the
container. The container has its own private network, and this is the door you open.

Worth doing while it runs, because it is the number that decides your Render plan:

```bash
docker stats --no-stream leaf-clinic
```

---

## Step 3 — Push to GitHub

```bash
git add .
git commit -m "Farmer assistant: CNN + RAG agent behind a FastAPI service"
git push -u origin main
```

**Check:** open your repo on GitHub. You should see `main.py`, `Dockerfile`,
`render.yaml` and the `.keras` file. You should **not** see `.env`. If you do,
delete the repo, rotate the key at Google AI Studio, and start again — a key in git
history is public forever, even after you delete the file.

---

## Step 4 — Deploy to Render

`render.yaml` in this repo is a **Blueprint**: it describes the service so you do
not have to configure it by hand in the dashboard.

1. Go to https://dashboard.render.com/blueprints and click **New Blueprint Instance**
2. Connect your GitHub account and pick this repository
3. Render reads `render.yaml` and shows you the service it is about to create:
   a Docker web service on the Standard plan, health check at `/health`
4. It will prompt you for **`GOOGLE_API_KEY`** — that is the `sync: false` line in
   the Blueprint doing its job, keeping the secret out of git. Paste your key.
5. Click **Apply** and watch the logs

The first build takes about ten minutes: Render downloads the base image, installs
TensorFlow, and copies the model in. Then startup takes another 30–60 seconds
before the health check passes.

**Check:** `https://YOUR-SERVICE.onrender.com/health` returns
`{"status": "ok", "model_loaded": true, ...}`. Then open the root URL on your phone,
over mobile data with wifi off. If it diagnoses a leaf, you have shipped.

### Doing it without the Blueprint

If you would rather click through it: **New → Web Service**, connect the repo,
choose **Docker** as the language, set the instance type to **Standard**, set the
health check path to `/health`, and add `GOOGLE_API_KEY` under **Environment**.
That is the same service `render.yaml` describes.

### After it is live

Every push to `main` redeploys automatically (`autoDeploy: true`). Turn it off in
the dashboard under **Settings → Build & Deploy** if you would rather deploy by hand.

To stop being billed between demos: **Settings → Suspend Web Service**. Resuming
takes one click and the next request pays the 30–60 second startup again.

---

## When it breaks

**Build fails on `pip install`** — a pinned version does not exist for Python 3.11.
Read which package it names and adjust that one line, not the whole file.

**`Exited with status 137` or "Ran out of memory (used over 512MB)"** — this is the
512 MB wall. Nothing is wrong with your code. Move to Standard, or go read
"Getting onto the free tier" below.

**Deploy hangs on "Port scan timeout, no open ports detected"** — the app is not
binding where Render is looking. It must be `0.0.0.0`, not `127.0.0.1`, and on
`$PORT`. That is what the `sh -c` form of `CMD` in the Dockerfile is for: the JSON
exec form would hand uvicorn the literal string `$PORT` instead of expanding it.

**`/health` returns 503 with an `error` field** — the app booted but `build_agent()`
failed, and the error text tells you which part. A missing `GOOGLE_API_KEY` is the
usual cause; the secret has to be named exactly that.

**`ValueError` when loading the model** — the TensorFlow version in
`requirements.txt` does not match the one that saved the file. Run
`import tensorflow as tf; print(tf.__version__)` in the notebook and pin that exact
version.

**`TypeError: Reviver.__init__() got an unexpected keyword argument 'allowed_objects'`**
— the LangChain packages have drifted apart. `langgraph` ships code that calls into
`langchain-core`, so a new `langgraph` with an old `langchain-core` fails at import.
This is why every LangChain package in `requirements.txt` is pinned together,
transitive ones included. Bump them as a set, never one at a time.

**First request after an idle period takes 30+ seconds** — only if you suspended the
service or are on a plan that spins down. Standard instances stay awake.

---

## Getting onto the free tier

Only TensorFlow is keeping this off a 512 MB instance. The fix is to stop shipping
a training framework to do inference:

1. In the training notebook, convert the saved model:
   ```python
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   open("plant_disease_model.tflite", "wb").write(converter.convert())
   ```
2. Drop `tensorflow-cpu` from `requirements.txt` and add `ai-edge-litert`.
3. In `agent.py`, replace `tf.keras.models.load_model` and `model.predict` with a
   LiteRT `Interpreter`, and replace `load_img`/`img_to_array` with Pillow plus
   NumPy — resize to 128×128, cast to float32, add a batch dimension.

Expect the service to land near 150 MB. The accuracy is unchanged; it is the same
weights. Budget an afternoon, and keep the TensorFlow version on a branch so you
can compare predictions between the two and confirm they agree.

---

## Put it on your résumé properly

Not "built a plant disease detection model". That sentence is on ten thousand résumés.

> **Tomato Leaf Clinic** — github.com/you/farmer-assistant-deploy · leafclinic.onrender.com
> Fine-tuned a CNN on PlantVillage tomato images (xx% validation accuracy) and served
> it with FastAPI behind a LangChain agent that routes between the classifier and a
> FAISS vector search over an agronomy guide, so treatment advice is retrieved rather
> than generated. Containerised with Docker and deployed to Render from a Blueprint.

Two links, one specific number, and a sentence that says what problem the
architecture solves. Fill in the accuracy from your own run — a made-up number is a
question you cannot answer in the interview.
