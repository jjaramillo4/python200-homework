# Week 6 Mini-Project: Groundwork Coffee Co. Q&A Assistant

# --- Step 1: Setup ---

import os
import string
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI as LlamaOpenAI  # judge LLM for Q4 (renamed to avoid clash)
from openai import OpenAI

env_path = Path(__file__).resolve().parent.parent / '.env'

load_dotenv(dotenv_path=env_path)

if os.getenv("OPENAI_API_KEY"):
    print("Successfully loaded API key")
else:
    print("OPENAI_API_KEY not found - check your .env file")


docs_dir = Path(__file__).resolve().parent / "groundwork_docs"
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"


# --- Step 2: Load the Documents ---

groundwork_docs = SimpleDirectoryReader(docs_dir).load_data()
print(f"Loaded {len(groundwork_docs)} Groundwork documents.")

for doc in groundwork_docs:
    print(doc.metadata["file_name"])

# --- Step 3: Build the Index and Query Engine ---
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = LlamaOpenAI(model="gpt-4o-mini")  # LlamaIndex's default (gpt-3.5-turbo) is retired

index = VectorStoreIndex.from_documents(groundwork_docs)

if index:
    print("Index built successfully. Ready to answer questions.")


query_engine = index.as_query_engine(similarity_top_k=3)

# --- Step 4: Query the Assistant ---
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    # Step 4 only asks for the top retrieved node (source_nodes are sorted best first)
    top_node = response.source_nodes[0]
    print(f"Top Source: {top_node.node.metadata['file_name']}")
    print(f"Similarity Score: {top_node.score:.4f}")
    print(f"Text Snippet: {top_node.node.get_content()[:200]}...")
    print("-" * 30)

# Reflection: all 5 answers were correct and sounded confident. The surprise was the
# weekend hours question: the top chunk was our_story.txt (no hours info), but the answer
# was still right because faq.txt was further down in the top 3. A correct answer doesn't
# mean retrieval ranked things right. The loyalty answer was also right despite a low
# score (0.28), since faq.txt covers many topics, which lowers its match to any one question.

# --- Step 5: Find a Failure ---

questions = [
    "Do you allow cats in yur establishment?",
    "What is your return policy? "
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    # Step 5 asks for all 3 retrieved nodes
    for node_with_score in response.source_nodes:
        print(f"Source: {node_with_score.node.metadata['file_name']}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
        print("-" * 30)

# What I asked: "Do you allow cats?" and "What is your return policy?". Neither is
# covered in the docs, so I expected the assistant to struggle.
#
# What went wrong: the return policy question was handled well, and it said there was no
# info. The cats question failed. faq.txt only has a dog policy (dogs on the patio, not
# inside). The model retrieved that chunk and stretched it into "Cats are not permitted
# inside" and "Only dogs are allowed on the patio", which the docs never say. It guessed.
#
# Tone: it still sounded confident even though it made up a cat policy, and the low scores
# (0.15-0.29) weren't shown to the user. You can't trust an answer just because it sounds sure.
#
# What I'd change: add a minimum similarity score cutoff so weak chunks are dropped, and
# tell the model in the prompt to only answer what the documents state, and to say
# "I don't know" otherwise.


# --- Step 6: Reflection ---
# 1) My LlamaIndex setup was only a few lines: load the docs, build the index, create the
#    query engine, and query it. The keyword RAG function from the lesson was ~40 lines,
#    and it didn't even do chunking, embeddings, or call an LLM. Doing all of that by hand
#    would take a lot more code. The value of a framework is that it's modular and handles
#    those steps for you, so you can quickly build a more complete and usable RAG system.
#
# 2) We actually have one at work that HQ built in Japan. It lets employees ask questions
#    about internal company information that a normal LLM wouldn't have access to, since
#    it was never in its training data. That being said, it's not perfect either.
#
# 3) Hallucinating, or making up information. Even when retrieval finds the right chunk,
#    the model can still misread it or stretch it too far, like in Step 5 where it turned
#    a dog policy into a cat policy. RAG gives the model the right sources, but it can't
#    force the model to stick to only what those sources say.





