import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pprint import pprint

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

print("Response: "+ response.choices[0].message.content)
print("Model: " + response.model)
print("Token Usage: "+ str(response.usage.total_tokens))

# API Question 2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures: 
    loop_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [{"role": "user", "content":prompt}],
        temperature = temp)
    print("Temperature: "+ str(temp))
    print("Response: "+ loop_response.choices[0].message.content)

# Temp of 0 gave answers that were predictable using the word Data. I would use if I needed a consistent mostly reproducible answer. 

# API Question 3

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

for index, choice in enumerate(response.choices, start=1) :
    print(str(index) + " Response: " + choice.message.content)

# API Question 4

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)
print("Limited 15 Max per Token: "+ response.choices[0].message.content)

#It would give a very vague answer or an answer that is cut off

# --- System Questions ---

# System Question 1

# System Question 2

# --- Prompt Questions ---

# Prompt Question 1

# Prompt Question 2

# Prompt Question 3

# Prompt Question 4

# Prompt Question 5

# Prompt Question 6

# --- Ollama ---

# Ollama Question 1