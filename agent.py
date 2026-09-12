"""
agent.py — everything that was in the Colab notebook, moved into a module.

The notebook ran top to bottom once. A server is different: it starts once and
then answers many requests. So the expensive work (loading the CNN, embedding
the guide, building the agent) happens exactly once inside build_agent(), and
every request afterwards just calls ask().
"""

import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.tools import tool
from langchain.agents import create_agent

IMG_SIZE = 128
MODEL_PATH = os.getenv("MODEL_PATH", "plant_disease_model.keras")
CLASSES_PATH = os.getenv("CLASSES_PATH", "class_names.json")
GUIDE_PATH = os.getenv("GUIDE_PATH", "plant_disease_guide.md")

SYSTEM_PROMPT = """You are a helpful assistant for farmers growing tomatoes.

Rules:
1. If the user gives an image file path, ALWAYS call detect_plant_disease first.
2. Then call search_treatment_guide using the detected disease name to get advice.
3. Base your treatment advice ONLY on what search_treatment_guide returns.
   Never invent a pesticide or a dosage.
4. If the plant is healthy, say so clearly and tell the farmer not to spray.

Answer in simple plain language a farmer can act on: what the disease is,
what to spray with the dosage, and how to prevent it next season."""


def build_agent():
    """Load every artifact and wire up the agent. Called once, at startup."""

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it as a secret where you deploy, "
            "or put it in a .env file when running locally."
        )

    # ---- 1. the CNN, loaded back from the file we saved in Colab -------------
    cnn_model = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.load(open(CLASSES_PATH))

    # ---- 2. the RAG pipeline ------------------------------------------------
    # In Colab we used a local sentence-transformer. Here we use Google's
    # embedding API instead: it removes torch and sentence-transformers from
    # the image, which takes the Docker build from ~3 GB down to ~1 GB.
    documents = TextLoader(GUIDE_PATH).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100
    ).split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # ---- 3. the two tools ---------------------------------------------------
    @tool
    def detect_plant_disease(image_path: str) -> str:
        """Identify the disease of a tomato plant from a leaf photo.
        Use this whenever the user provides a path to an image file.
        Input must be the full file path, for example /tmp/leaf.jpg
        Returns the predicted disease name and the confidence score."""
        image = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        batch = np.expand_dims(img_to_array(image), axis=0)
        probs = cnn_model.predict(batch, verbose=0)[0]
        best = int(np.argmax(probs))
        return f"Predicted disease: {class_names[best]} (confidence {probs[best]:.1%})"

    @tool
    def search_treatment_guide(query: str) -> str:
        """Search the official tomato disease guide for symptoms, pesticides, dosage,
        treatment steps and prevention advice. Use this for any question about how to
        treat, spray or prevent a disease. Input should be a disease name or a
        description of the symptoms."""
        query = query.replace("___", " ").replace("_", " ")
        results = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in results)

    # ---- 4. the agent -------------------------------------------------------
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    agent = create_agent(
        model=llm,
        tools=[detect_plant_disease, search_treatment_guide],
        system_prompt=SYSTEM_PROMPT,
    )

    return agent, detect_plant_disease, class_names


def ask(agent, question: str) -> dict:
    """Run the agent once and return the answer plus which tools it used.

    The tool list is worth returning: it is what lets the web page show
    'the agent looked at your photo, then searched the guide' instead of
    a black box.
    """
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    tools_used = []
    for message in result["messages"]:
        for call in getattr(message, "tool_calls", None) or []:
            tools_used.append(call["name"])

    return {"answer": result["messages"][-1].text, "tools_used": tools_used}
