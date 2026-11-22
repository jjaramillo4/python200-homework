# Week 6 Warmup Exercises

# --- Setup ---

import os
import textwrap
from pathlib import Path
from dotenv import load_dotenv
from pprint import pprint
import json
import string
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI  # judge LLM for Q4 (renamed to avoid clash)
from openai import OpenAI

env_path = Path(__file__).resolve().parent.parent / '.env'

load_dotenv(dotenv_path=env_path)

if os.getenv("OPENAI_API_KEY"):
    print("Successfully loaded API key")
else:
    print("OPENAI_API_KEY not found - check your .env file")


# --- RAG Concepts ---

# Concepts Q1
# Scenario A (Legal team, policy library) -> RAG
#   The legal team has hundreds of PDFs that are updated every quarter, so RAG is the
#   best fit. It lets the model pull answers from the current documents, and when a
#   policy changes they just swap in the new PDF instead of retraining anything.
#
# Scenario B (Startup, brand voice) -> Fine-tuning
#   They need the model to learn a style, not look up facts. The brand voice is
#   consistent and doesn't change, and with 3,000 examples they have plenty of data
#   to train the model on that voice so it's more relevant to what they need.
#
# Scenario C (Analyst, single two-page report) -> Prompt engineering
#   It's one short document that she only needs once, so she can just paste or upload
#   the report into the prompt and ask the model to answer using it. Building a RAG
#   pipeline or fine-tuning would be overkill.


# Concepts Q2
# Hallucinations are a real problem for LLMs, and how often they happen can depend on
# how the model was trained and the role it's given. A confidently wrong answer is more
# harmful than "I am not sure" because nothing signals to the user that they should
# double-check it. An "I'm not sure" tells them to go verify, while a confident answer
# gets taken at face value.
#
# Example: in a medical situation, a user describes their symptoms and the model
# confidently gives the wrong diagnosis or treatment advice. If it sounds sure, the
# person might skip seeing a doctor or take the wrong medication.
#
# Tone matters because we naturally trust answers that sound certain and specific.
# But the model isn't self-aware: it's just predicting the next most likely words,
# so it sounds equally confident whether it's right or wrong. Its confidence doesn't
# tell you anything about its accuracy.


# Concepts Q3
# steps = [
#    "Extract text from source documents",       # Pull the raw text out of the files.
#    "Split text into chunks",                   # Break the text into small pieces.
#    "Convert text chunks into embeddings",      # Turn each chunk into a vector of numbers.
#    "Receive the user's query",                 # The user asks a question.
#    "Embed the user's query",                   # Turn the question into a vector too.
#    "Retrieve the most relevant chunks",        # Find the chunks most similar to the question.
#    "Inject retrieved chunks into the prompt",  # Add those chunks to the prompt as context.
#    "Generate a response from the LLM",         # The LLM answers using that context.
# ]

# --- Keyword RAG ---

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


# Keyword Q1
query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

result=simple_keyword_retrieval(query,documents, verbose=True)

print(result[0][0])
print(result[0][1])

# loyalty.txt was selected, which is wrong. hours.txt, hiring.txt, and loyalty.txt
# all tied with 1 overlapping word. hours.txt matched "weekends" (the useful word),
# but hiring.txt and loyalty.txt matched "your", a filler word that isn't in the
# stopword list. On a tie, sort(reverse=True) compares the names in reverse
# alphabetical order, so loyalty.txt won. Keyword matching can't tell important
# words from filler words, and "hours" didn't match because hours.txt never uses that word.

# Keyword Q2
query = "Do you have anything without caffeine?"

result = simple_keyword_retrieval(query,documents, verbose =True)
print(result[0][0])
print(result[0][1])

# No document was selected: every document scored 0, so it returned "None found".
# Keyword RAG got this wrong. menu.txt is the right document (it lists coffee drinks
# and pastries), but it never uses the words "caffeine", "without", or "anything",
# so there was no exact word match. Semantic retrieval with embeddings would do better
# because it compares meaning, so it could connect "caffeine" to coffee drinks.

# Keyword Q3
# Prediction: no match. loyalty.txt is the right document, but it says "join",
# "points", and "redeem" instead of "sign up" and "rewards", so no words overlap.
query = "How do I sign up for rewards?"
result = simple_keyword_retrieval(query,documents, verbose =True)
print(result[0][0])
print(result[0][1])

# Result: my prediction was correct. Every document scored 0 and nothing was selected.
# Same synonym problem as Q2: keyword RAG can't match words with the same meaning.

# --- Semantic RAG Concepts ---

# Semantic Q1
# 1) A vector embedding turns a piece of text into a list of numbers that represents
#    its meaning. Texts with similar meanings get similar numbers, so they end up
#    close together.
#
# 2) The chunk with 0.85 is more relevant. Cosine similarity goes up to 1, and the
#    closer to 1, the more similar the meaning. 0.85 means that chunk is very close
#    in meaning to the query, while 0.30 means it's only loosely related.
#
# 3) Semantic search compares meaning instead of exact words. Similar ideas have
#    similar embeddings, so a chunk can match even if it uses different words
#    (like "caffeine" matching a chunk about espresso and cold brew).


# Semantic Q2
# | Feature                    | Keyword RAG                       | Semantic RAG                           |
# |----------------------------|-----------------------------------|----------------------------------------|
# | What is compared?          | Exact word overlap                | Meaning (embedding vectors)            |
# | What is retrieved?         | Full document                     | The most relevant chunks               |
# | Can it handle synonyms?    | No                                | Yes                                    |
# | Storage format             | Plain text dictionary             | Vector store / index of embeddings     |
# | Relevance score            | Number of overlapping keywords    | Cosine similarity score                |


# --- LlamaIndex ---

# LlamaIndex Setup (embedding model)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = LlamaOpenAI(model="gpt-4o-mini")  # LlamaIndex's default (gpt-3.5-turbo) is retired
brightleaf_path = Path(__file__).resolve().parent / "brightleaf_pdfs"
brightleaf_docs = SimpleDirectoryReader(brightleaf_path).load_data()
print(f"Loaded {len(brightleaf_docs)} Brightleaf documents.")



# LlamaIndex Q1

index = VectorStoreIndex.from_documents(brightleaf_docs)

query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

# Benefits query:
#   - Relevant? The top chunk was the employee benefits doc (0.76), which is right.
#     The other two (EcoVolt partnership, company overview) were only loosely related.
#   - Tone: very confident and specific (Blue Cross, 5% 401(k) match, 12 weeks parental
#     leave). No hedging like "based on the context".
#   - Unexpected: the partnership doc showed up even though it has nothing to do with benefits.
#
# Security query:
#   - Relevant? The top chunk was the security policy doc (0.68), which is right.
#     The other two (employee benefits, company overview) were not really relevant.
#   - Tone: also very confident and detailed (MFA, TLS 1.3, AES-256, NIST 800-61). No hedging.
#   - Unexpected: the benefits doc was retrieved for a security question. With only 6 docs,
#     top_k=3 always returns 3 chunks, so it fills the extra slots with weak matches.

# LlamaIndex Q2
query_engine = index.as_query_engine(similarity_top_k=1)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

query_engine = index.as_query_engine(similarity_top_k=5)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

# top_k=1 vs top_k=5: the answers barely changed. top_k=1 retrieved only the right doc
# (benefits 0.76, security 0.68), and that one chunk already had everything needed.
# top_k=5 added 4 more chunks with lower scores (down to ~0.43) from unrelated docs like
# the EcoVolt partnership and the company overview, but the answer stayed about the same.
# More context isn't always better: the extra chunks were mostly noise, cost more tokens,
# and could distract the model or mix in wrong info. It only helps when the answer is
# spread across several documents.


# LlamaIndex Q3

query_engine = index.as_query_engine(similarity_top_k=5)

questions = [
    "What is the company's dress code?",
    "What is the policy on remote and hybrid work?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

# Dress code (not in the documents):
#   - Expected: the model to guess a dress code since it sounds like a normal HR question.
#   - What happened: it correctly said the info isn't in the documents. All 5 retrieved
#     chunks had very low scores (0.13-0.28), so the retriever found nothing relevant.
#   - Note: it still returned 5 chunks anyway, because top_k always returns k results.
#
# Remote/hybrid work (only briefly mentioned):
#   - Expected: a weak answer since there's no real remote work policy, just one line in
#     the benefits doc.
#   - What happened: it answered confidently from that one benefits sentence and made it
#     sound like a full policy. Scores were low (0.22-0.37), but the tone didn't show that.
#
# What I'd change: add a minimum similarity score cutoff so weak chunks get dropped, and
# tell the model in the prompt to say "I don't know" when the context doesn't cover it.


# LlamaIndex Q4

llm = LlamaOpenAI(model="gpt-4o-mini")

# Q4 says to use the Q1 query engine, so reset it to top_k=3 (Q3 left it at 5)
query_engine = index.as_query_engine(similarity_top_k=3)

faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

q = "What employee benefits does BrightLeaf offer?"
response = query_engine.query(q)

faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))

q = "What is the dress code for the company?"
response = query_engine.query(q)

faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))

# Results: benefits query scored 1.0 / 1.0, dress code query scored 0.0 / 0.0.
#
# - Faithfulness 1.0 means everything in the answer is backed up by the retrieved
#   chunks (no made-up info). 0.0 means the answer isn't supported by the chunks.
# - Relevancy measures whether the answer and retrieved chunks actually address the
#   question. Faithfulness checks "is it grounded in the sources?", relevancy checks
#   "does it answer what was asked?".
# - Yes, the scores dropped to 0 for the dress code question. The documents don't
#   cover it, so the retrieved chunks were unrelated and the answer ("no info on a
#   dress code") couldn't be supported by them or answer the question.
# - LLM-as-a-judge means using a second LLM to grade the first one's answer. It's used
#   because RAG answers are open-ended text with no single correct answer to compare
#   against, so a simple accuracy metric doesn't work.