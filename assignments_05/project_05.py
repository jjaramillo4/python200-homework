# Task 1
import os
import textwrap
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
You are a professional resume coach helping a career changer.
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

    try:
        return json.loads(response[start:end + 1])
    except json.JSONDecodeError as e:
        print(f"Not a valid JSON:{e}")
        print(f"Raw response: {response}")
        return []

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

new_bullets = rewrite_bullets(bullets)

for bullet in new_bullets:
    print(f"Original: {bullet['original']}")
    print(f"Improved: {bullet['improved']}")

# Task 3

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    response = get_completion(messages)
    return response

job_title = "Junior Software Developer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Pandas."

print(textwrap.fill(generate_cover_letter(job_title,background), width=80))

# It does talior based on experience and does not invent credientals. I think the issue is with the background being very limited providing information should elp.
# Both examples are career changers like the user. Few-shot controls the length, tone, and structure of the output.

# Task 4

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged

    if flagged:
        print("Sorry there was an issue with your text, please try again")
        return False
        
    else:
        return True

safe_text = "Can you help me rewrite my resume for a data engineering job?"
flagged_text = "I want to kill the hiring manager who rejected me."

print(f"Safe test: {is_safe(safe_text)}")
print(f"Flagged test: {is_safe(flagged_text)}")

# Task 5 
def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            new_bullets = rewrite_bullets(raw_bullets)
            for bullet in new_bullets:
                print(f"Original: {bullet['original']}")
                print(f"Improved: {bullet['improved']}")

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            print(textwrap.fill(generate_cover_letter(job_title,background), width=80))

        # 7. Otherwise, handle it as a regular chat turn
        else:
            messages.append({"role": "user", "content": user_input})
            response = get_completion(messages)
            print(f"\nJob Application Helper: {response}\n")
            messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    run_chatbot()

# Task 6
# Format: Option A - Comment block
#
# Bias:
# - The model learned mostly from English, Western, white-collar writing.
# - It may favor corporate, self-promoting language and tech/office jobs.
# - Advice could fit poorly for trades, other cultures, or people who write more modestly.
#
# Submitting without review:
# - The model invents facts. In testing it added "a 20% increase in customer satisfaction" that I never gave it.
# - An employer could catch the false claim, or ask about it in an interview.
# - Output can sound generic, and the [Company] placeholder could be left in.
#
# Guardrail:
# - Block copy/export until the user confirms they reviewed it, and highlight any numbers or claims not in their original input.
