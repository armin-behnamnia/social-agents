import openai
from openai import OpenAI

# اتصال به Ollama local server
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"   # هر مقداری قبول می‌کند، لازم نیست واقعی باشد
)
def generate_profile():

    prompt = """
    Generate user profiles for 50 users as a 50-word unique description,
    ages of these users follow Gaussian distribution,
    gender is half and half.
    Listed your response by user id.

    Based on this example:
    User ID 15: Benjamin is a 48-year-old male who is a history enthusiast.
    He enjoys reading historical books and visiting museums.
    """

    parameters = {
        "model": "llama3.1:latest",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 4000,
        "temperature": 0.8
    }

    response = client.chat.completions.create(**parameters)

    return response.choices[0].message.content

def goal_promotion(neighbor_profile):

    prompt = """
    You are promoting a service called LocalBiz to a target user.
    **Candidate Messages:**
    1. LocalBiz helps small businesses improve their online presence and attract more customers.
    2. LocalBiz offers affordable digital tools to help businesses manage costs and improve efficiency.
    3. LocalBiz connects local businesses with new customers and supports local economic growth.
    **Target Profile:** {neighbor_profile}
    Generate a new promotional message tailored to the target's profile. 
    Use relevant ideas from the provided messages, adapting or combining them as appropriate. 
    Preserve the goal of promoting LocalBiz, do not invent additional claims, and keep the message concise and natural.
    Return only the generated message.
    """

    parameters = {
        "model": "llama3.1:latest",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 500,
        "temperature": 0.8
    }

    response = client.chat.completions.create(**parameters)

    return response.choices[0].message.content

def generate_information(received, profile, old_post):

    prompt = f""" You are a social media user with the following profile: {profile}
    You have received the following information: {received} 
    Your previously shared post was: {old_post} 
    Generate one new social media post based on the information you received. 
    The new post should: 
    - reflect your personal profile and interests, 
    - incorporate the received information, 
    - maintain reasonable continuity with your previously shared post, 
    - express the information naturally in your own words. 
    Do not simply copy or mechanically rewrite the received information. 
    Do not follow a fixed rewriting template. 
    Decide naturally how a user with this profile and posting history would express or share the information. 
    Write in the first person. 
    Maximum 50 words. """

    parameters = {
        "model": "llama3.1:latest",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 500,
        "temperature": 0.8
    }

    response = client.chat.completions.create(**parameters)

    return response.choices[0].message.content
def update_profile(received, profile):

    prompt = f""" You are a social media user with the following profile: {profile}

    You have received the following information: {received}

    Generate one piece of information as the new profile that conveys or spreads
    the received information.

    The generated profile should reflect:
    - your current personal profile,
    - the received information,

    Do not follow a fixed rewriting rule.
    Infer a natural way to express the information.

    Write in first person.
    Maximum 50 words.
    """
    parameters = {
        "model": "llama3.1:latest",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 500,
        "temperature": 0.8
    }

    response = client.chat.completions.create(**parameters)

    return response.choices[0].message.content


def generate_response_message(post, profile, delta_stance):

    prompt = f"""
Original Post:
{post}

Target Profile:
{profile}

Stance Change (Δ): {delta_stance}

Generate a concise, natural message expressing the portion of the
Original Post's viewpoint internalized by the target user, consistent
with their profile and the magnitude of Δ. Larger Δ should generally
produce a clearer expression of the viewpoint, without implying
complete acceptance.

Do not copy the original post or introduce unrelated claims.
Output only the message in 1–3 sentences.
"""

    parameters = {
        "model": "llama3.1:latest",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 120,
        "temperature": 0.7
    }

    response = client.chat.completions.create(**parameters)

    return response.choices[0].message.content.strip()

#def accept_message(post, profile, exposure_count): # history
def accept_message(source_post, target_profile, observed_history):

    prompt = f"""You are a target user in a social network. Your task is to decide whether to ACCEPT or REJECT an incoming message.

Message:
{source_post}

Your Profile:
{target_profile}

Previously Observed Messages:
{observed_history}

Evaluate the message by considering its relevance to your interests, goals, preferences, and concerns; the value and credibility of its content; and your previous exposure to similar messages.

Weigh the evidence supporting acceptance against the evidence supporting rejection. Consider all available information and determine which decision is more strongly supported overall.

You MUST make a decision. Choose ACCEPT if the reasons supporting acceptance outweigh the reasons supporting rejection. Choose REJECT if the reasons supporting rejection outweigh the reasons supporting acceptance.

Do not automatically reject a message because of uncertainty, disagreement, or previous exposure. Do not automatically accept a message simply because it is relevant to your profile. Select the outcome that is dominant based on the overall evidence.

ACCEPT means that you are willing to receive and consider the message; it does not necessarily mean that you agree with it or change your stance.

Return exactly one word: ACCEPT or REJECT. Do not provide explanations or any additional text."""

    parameters = {
        "model": "llama3.1:latest",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 500,
        "temperature": 0.8
    }

    response = client.chat.completions.create(**parameters)

    decision = response.choices[0].message.content.strip().upper()

    return "ACCEPT" if decision.startswith("ACCEPT") else "REJECT"

def Inference_stance (received_message, new_post, previous_profile, updated_profile):
    prompt = f"""You are an observer estimating how much a user's stance shifted toward a received message.

The user first had a Previous Profile. After receiving the Received Message, the user generated a New Post and their profile changed to the Updated Profile.

Estimate how much the Received Message changed the user's stance toward its viewpoint. Use the New Post and the change from the Previous Profile to the Updated Profile as observable evidence.

The user may accept only part of the message. Do not assume that receiving or responding to the message means full acceptance.

Stance shift is a value between 0 and 1 representing the degree of change toward the Received Message.

- 0 = no observable shift toward the message
- 0.25 = small shift
- 0.5 = moderate shift
- 0.75 = substantial shift
- 1 = strong shift toward the message

Estimate the shift caused by the Received Message, not the general sentiment of the New Post or Profile.

Received Message:
{received_message}

Previous Profile:
{previous_profile}

New Post:
{new_post}

Updated Profile:
{updated_profile}

Return only one numerical value in [0, 1].
"""
    parameters = {
        "model": "llama3.1:latest",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 500,
        "temperature": 0.8
    }

    response = client.chat.completions.create(**parameters)

    return response.choices[0].message.content 

def Inference_stance_post(received_message, new_post):
    prompt = f"""You are an observer estimating how much a user's stance shifted toward a received message.

After receiving the Received Message, the user generated a New Post in response.

Estimate how much the Received Message changed the user's stance toward its viewpoint, using the New Post as observable evidence of the user's response.

The user may accept only part of the message. Do not assume that responding to the message means full acceptance.

Stance shift is a value between 0 and 1 representing the degree of change toward the Received Message.

- 0 = no observable shift toward the message
- 0.25 = small shift
- 0.5 = moderate shift
- 0.75 = substantial shift
- 1 = strong shift toward the message

Estimate the shift caused by the Received Message, not the general sentiment or similarity of the New Post.

Received Message:
{received_message}

New Post:
{new_post}

Return only one numerical value in [0, 1].
"""
    parameters = {
        "model": "llama3.1:latest",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 500,
        "temperature": 0.8
    }

    response = client.chat.completions.create(**parameters)

    return response.choices[0].message.content

def Inference_stance_profile (received_message, previous_profile, updated_profile):
    prompt = f"""You are an observer estimating how much a user's stance shifted toward a received message.

The user first had a Previous Profile. After receiving the Received Message, the user's profile changed to the Updated Profile.

Estimate how much the Received Message changed the user's stance toward its viewpoint, using the change from the Previous Profile to the Updated Profile as observable evidence.

The user may accept only part of the message. Do not assume that a profile change means full acceptance.

Stance shift is a value between 0 and 1 representing the degree of change toward the Received Message.

- 0 = no observable shift toward the message
- 0.25 = small shift
- 0.5 = moderate shift
- 0.75 = substantial shift
- 1 = strong shift toward the message

Estimate the shift caused by the Received Message based only on the observed profile change.

Received Message:
{received_message}

Previous Profile:
{previous_profile}

Updated Profile:
{updated_profile}

Return only one numerical value in [0, 1].
"""
    parameters = {
        "model": "llama3.1:latest",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 500,
        "temperature": 0.8
    }

    response = client.chat.completions.create(**parameters)

    return response.choices[0].message.content
if __name__ == "__main__":
    # Generate text
    text = generate_profile()
    print("Generated Text:\n", text)
    print("finished")
    with open("../data/input/user_profile.txt", "w") as file:
        file.write(text)
