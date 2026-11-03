# Task 1
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

client = OpenAI()

SYSTEM_PROMPT = """
Stay focused on job application materials. redirect the user back to the task if they go off-topic
Always remind the user to review and edit their output before submitting anywhere
Acknowledge that it may not know the user's specific industry norms, and that the user should use their own judgment
Do not invent credentials, skills, or experiences for the user. 
Ask before guessing, ask for clarification, and ask for more information if needed.
Tone should be blunt, professional, and concise. Avoid being overly verbose or flowery.
Do not write the whole application for the user. Instead, provide guidance, suggestions, and examples to help the user improve their own work.
"""
def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "Help me with my resume."}]
print(get_completion(messages))

### Added statements to keep it focused on the task and to remind the user to review and edit their output before submitting anywhere. Also, added a note that the model may not know the user's specific industry norms, and that the user should use their own judgment. Finally, added a note that the model should not invent credentials, skills, or experiences for the user, and should ask for clarification if needed.

# Task 2

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).
    Do not wrap the JSON in markdown code fences and do not add any text before or after it.

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]

    response = get_completion(messages)

    # The model sometimes wraps the JSON in a markdown fence or adds a sentence
    # before it, so take only the text between the first [ and the last ].
    start = response.find("[")
    end = response.rfind("]")
    if start == -1 or end == -1:
        print(f"Could not find JSON in the response:\n{response}")
        return []

    return json.loads(response[start:end + 1])

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

print(rewrite_bullets(bullets))

# Task 3

# Task 4

# Task 5 

# Task 6
