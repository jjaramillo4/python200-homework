import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pprint import pprint
import json 

env_path = Path(__file__).resolve().parent.parent / '.env'

load_dotenv(dotenv_path=env_path)

if os.getenv("OPENAI_API_KEY"):
    print("Successfully loaded API key")
else:
    print("OPENAI_API_KEY not found - check your .env file")

# --- Completions API ---

# API Question 1
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [{"role": "user", "content":"What is one thing that makes Python a good language for beginners?"}]
)

print(f"Response: {response.choices[0].message.content}")
print(f"Model: {response.model}")
print(f"Token Usage: {response.usage.total_tokens}")

# API Question 2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures: 
    loop_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [{"role": "user", "content":prompt}],
        temperature = temp)
    print(f"Temperature: {temp}")
    print(f"Response: {loop_response.choices[0].message.content}")

# Temp of 0 gave answers that were predictable using the word Data. I would use it if I needed a consistent mostly reproducible answer.
# TODO: the question asks how the outputs DIFFER -- add a line on what 0.7 and 1.5 produced.

# API Question 3

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

for index, choice in enumerate(response.choices, start=1) :
    print(f"{index} Response: {choice.message.content}")

# API Question 4

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_completion_tokens=15
)
print(f"Limited 15 Max per Token: {response.choices[0].message.content}")

# It would give a very vague answer or an answer that is cut off. This would be useful if you wanted to limit the amount of text returned to save on tokens or if you wanted to force the model to give a very concise answer.

# --- System Questions ---

# System Question 1

response = client.chat.completions.create(
    model="gpt-4o-mini",
  messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
)

print(f"System Question 1: {response.choices[0].message.content}")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
    {"role": "system", "content": "You are an impatient Data Engineer. You always explain sources to find information, but do not answer directly."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
)

print(f"System Question 1 Changed role: {response.choices[0].message.content}")

# The model would not answer the question directly, but instead would provide sources to find information about list comprehensions. This would be useful if you wanted to encourage the user to do their own research and learn from multiple sources.

# System Question 2

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]
)

print(f"System Question 2: {response.choices[0].message.content}")
# The model would remember the user's name from the previous message and respond with "Your name is Jordan." This would be useful if you wanted to create a more personalized experience for the user.

# --- Prompt Questions ---

# Prompt Question 1
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for index, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Please summarize the following reviews as positive, negative, or mixed: '{review}'"}]
    )
    print(f"Prompt Question 1 - Review {index} Sentiment: {response.choices[0].message.content}")

# Prompt Question 2

for index, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"""Please summarize the following reviews as positive, negative, or mixed:

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Review: "{review}"
Sentiment:"""}]
    )
    print(f"Prompt Question 2 - Review {index} Sentiment: {response.choices[0].message.content}")

# Yes the format changed and it was more consistent, but because mixed was the example it leaked the label. 

# Prompt Question 3
for index, review in enumerate(reviews, start=1):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"""Please summarize the following reviews as positive, negative, or mixed:

Example:
Review: "Fast shipping and great customer service."
Sentiment: positive

Example:
Review: "Slow shipping and poor customer service."
Sentiment: negative

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Review: "{review}"
Sentiment:"""}]
    )
    print(f"Prompt Question 3 - Review {index} Sentiment: {response.choices[0].message.content}")

# one-shot uses less tokens but is not formatted as well. The few-shot prompt is more consistent and formatted better, but uses more tokens. I would use the few-shot prompt if I wanted a more consistent and formatted output, but I would use the one-shot prompt if I wanted to save on tokens.

# Prompt Question 4
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
        {"role": "user", "content": """A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later takes a new job that pays $7,500 more per year than her post-raise salary. What is her final annual salary?

Please show your work and explain your reasoning step by step, and label the final answer.
"""}],
 temperature=0
)

print(f"Prompt Question 4: {response.choices[0].message.content}")

# Step-by-step reasoning is important for complex problems, as it allows the model to break down the problem into manageable parts and ensure that each step is logically sound. This can help prevent errors and provide a clear explanation of how the final answer was reached.

# Prompt Question 5
review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": f"""Analyze the review below and return ONLY valid JSON with exactly these keys:
"sentiment" (positive, negative, or mixed), "confidence" (a float from 0 to 1), and "reason" (one sentence).

Review: {review}"""}]
)

raw = response.choices[0].message.content
print(f"Prompt Question 5 - Raw response: {raw}")

try:
    data = json.loads(raw)
    print(f"Prompt Question 5 - Sentiment: {data['sentiment']}")
    print(f"Prompt Question 5 - Confidence: {data['confidence']}")
    print(f"Prompt Question 5 - Reason: {data['reason']}")

except json.JSONDecodeError as e:
    print(f"Not a valid JSON:{e}")
    print(f"Raw response: {raw}")

# Prompt Question 6
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

second_user_text = "I promise you, you do not need it. \
I know it seems like a good idea, but it is not. "

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

second_prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{second_user_text}```
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]  
)

print(f"Prompt Question 6 - User Text Response: {response.choices[0].message.content}")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": second_prompt}]  
)

print(f"Prompt Question 6 - No Steps Provided Response: {response.choices[0].message.content}")

# Back ticks are used to format text as code or preformatted text in markdown. In this case, they are used to clearly delineate the user-provided text from the rest of the prompt, making it easier for the model to identify and process the instructions within that text.

# --- Ollama ---

# Ollama Question 1

"""
Thinking...
Okay, the user wants me to explain a large language model in two sentences. Let me start by recalling what I know. Large language models are big artificial intelligence models that can 
process and understand text, right? They're used in various applications like chatbots, translation, and research. But how to structure that in two sentences?

First sentence: Introduce the concept, maybe mention their size and capabilities. Second sentence: Highlight their use cases and impact on different fields. Need to make sure it's concise 
and covers both aspects. Let me check if I'm not missing anything. Oh, also, maybe mention that they're trained on massive datasets. Yes, that's important. Alright, putting it all together.
...done thinking.

A large language model is an artificial intelligence system trained on vast amounts of text to understand and generate human-like language. It enables complex tasks such as writing,
translation, and answering questions, making it a powerful tool for various industries.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain what a large language model is in two sentences."}]
)

print(f"Ollama Question 1 - OpenAI Response: {response.choices[0].message.content}")

# Differences: The OpenAI response was more detailed. It named the underlying techniques (deep learning, neural networks)
# and gave more example tasks, while the local qwen3 answer stayed general. qwen3 also printed a "Thinking..." block
# before its answer, which gpt-4o-mini did not.

# Advantages: Local models can offer better privacy and control over data, as they don't rely on external servers.
# They can also be optimized for specific tasks or domains, potentially improving performance, and run locally on your pc.

# Disadvantages: Local models may require significant computational resources, which can be a barrier for some users.
# They might also lack the latest updates or improvements found in cloud-based models.