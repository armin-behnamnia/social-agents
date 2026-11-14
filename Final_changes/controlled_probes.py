"""Controlled susceptibility probes, separate from natural diffusion.
A caller supplies a decision callback backed by the SAME acceptance gate used
in reference diffusion. Each callback receives an isolated deep-copy; results
must never be written back to the live simulation. Predictions precede labels.
"""
from copy import deepcopy
import random


def run_probes(profiles, messages, predict, decide, pair_id, seed=2026):
    rng = random.Random(seed)
    trials = [(profile, message) for profile in profiles for message in messages]
    rng.shuffle(trials)
    rows = []
    for index, (profile, message) in enumerate(trials):
        public = dict(profile_text=profile['public_text'], message_text=message['text'])
        forecasts = predict(deepcopy(public))
        # The decision callback may receive hidden profile state; predictor may not.
        decision = decide(deepcopy(profile), deepcopy(message))
        if decision not in ('ACCEPT', 'REJECT'):
            raise ValueError('Decision must be ACCEPT or REJECT')
        rows.append(dict(pair_id=pair_id, probe_id=f'{pair_id}:{index}',
                         profile_id=profile['id'], message_id=message['id'],
                         **public, predictions=forecasts, label=int(decision == 'ACCEPT')))
    return rows
