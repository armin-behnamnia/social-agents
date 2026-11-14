"""Optional public-text stance measurement; no hidden state arguments."""
import json


def estimate(text, topic):
    from source.LLM_module import client
    response = client.chat.completions.create(
        model='llama3.1:latest', temperature=0,
        messages=[{'role': 'user', 'content':
            f'Estimate the author stance toward this fixed proposition: {topic}. '
            'Use only the public text below. Return JSON with stance in [-1,1] '
            '(opposition to support) and uncertainty in [0,1]. '
            'For insufficient evidence use stance=0, uncertainty=1. Text:\n' + str(text)}])
    data = json.loads(response.choices[0].message.content)
    stance, uncertainty = float(data['stance']), float(data['uncertainty'])
    if not -1 <= stance <= 1 or not 0 <= uncertainty <= 1:
        raise ValueError('Invalid observer output')
    return stance, uncertainty
