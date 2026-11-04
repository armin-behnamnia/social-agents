"""Paired-run state management and leakage-separated evaluation outputs."""
import copy
import json
import os
import random
import uuid

import numpy as np
import pandas as pd

# Explicit whitelist: never deepcopy framework/model/network references.
STATE_FIELDS = (
    'preference_state isSeed profile stance stance_label support_probability post '
    'internalized emerged embedding profile_embedding post_embedding internalized_embedding '
    'exposure_count num_infected_neighbors new_adopter adoption_time degree '
    'degree_centrality betweenness_centrality message_id parent_message_id '
    'original_message_id received_message_history generated_message_history '
    'behavior_history interaction_history accept accept_embedding '
    'decision_history topic_exposures dominant_decision_style'
).split()


class EvaluationMixin:
    def capture_initial_state(self, agents):
        self._initial_agents = {
            a.id: {k: copy.deepcopy(getattr(a, k)) for k in STATE_FIELDS if hasattr(a, k)}
            for a in agents
        }
        self._initial_messages = copy.deepcopy(self.message_events)
        self._initial_observer = copy.deepcopy(self.observer_states)
        self._python_state = random.getstate()
        self._numpy_state = np.random.get_state()
        self.pair_id = uuid.uuid4().hex

    def begin_condition(self, agents, condition):
        if not hasattr(self, '_initial_agents'):
            raise RuntimeError('Capture the seeded initial state before running conditions.')
        for a in agents:
            state = self._initial_agents[a.id]
            for key in STATE_FIELDS:
                if key in state:
                    setattr(a, key, copy.deepcopy(state[key]))
                elif hasattr(a, key):
                    delattr(a, key)
            a.influencer = None
        random.setstate(self._python_state)
        np.random.set_state(self._numpy_state)
        self.observer_rng = np.random.default_rng(getattr(self.scenario, 'random_seed', 2026))
        self.observer_states = copy.deepcopy(self._initial_observer)
        self.condition = condition
        self.run_id = self.pair_id + ':' + condition
        self.diffusion_events = []
        self.message_events = copy.deepcopy(self._initial_messages)
        for row in self.message_events:
            row.update(pair_id=self.pair_id, run_id=self.run_id, condition=condition,
                       period=-1, decision='SEED')
            row['generated_post'] = row.pop('generated_message', row.get('generated_post'))
        self.observable_events = []
        self.prediction_labels = []
        self.trajectories = []
        self.candidate_snapshots = []
        self.transmissions = []
        self._period_index = 0
        self.record_trajectory(agents, -1, 'initial')

    def reset(self, agents):
        self.begin_condition(agents, 'stance_less')

    def record_trajectory(self, agents, period, phase='end'):
        for a in agents:
            state = self.observer_states[a.id]
            self.trajectories.append(dict(
                pair_id=self.pair_id, run_id=self.run_id, condition=self.condition,
                period=period, phase=phase, agent_id=a.id, stance=a.stance,
                estimated_stance=state['estimated_stance'],
                state_variance=state['state_variance'],
                adopted=int(a.preference_state == 1), is_seed=int(a.isSeed),
                adoption_time=a.adoption_time, exposure_count=a.exposure_count,
                profile=a.profile, post=a.post, message_id=a.message_id,
            ))

    def agents_infection_stance_aware(self, agents, period, k=1):
        return self._diffusion_step(agents, period, True)

    def agents_infection(self, agents, period, k=1):
        return self._diffusion_step(agents, period, False)

    def public_features(self, agent):
        state = self.observer_states[agent.id]
        stance, uncertainty = state['estimated_stance'], state['state_variance']
        text = str(agent.profile) + '\n' + str(agent.post or '')
        if getattr(self.scenario, 'text_stance_observer', False):
            from source.public_stance import estimate
            stance, uncertainty = estimate(text, self.scenario.stance_topic)
        return dict(profile_text=text, estimated_stance=float(stance),
                    stance_uncertainty=float(uncertainty),
                    degree_centrality=agent.degree_centrality,
                    betweenness_centrality=agent.betweenness_centrality)

    def _diffusion_step(self, agents, period, stance_aware):
        self._history_agents = agents
        for agent in agents:
            if agent.preference_state != 1 and not agent.isSeed:
                self.candidate_snapshots.append(dict(
                    snapshot_id=f'{self.run_id}:{period}:{agent.id}',
                    run_id=self.run_id, pair_id=self.pair_id, period=period,
                    agent_id=agent.id, **self.public_features(agent)))
        # Snapshot BEFORE any exposure: newly adopted nodes wait until next period.
        sources = [a for a in agents if a.preference_state == 1 and a.post and str(a.post).strip()]
        for a in agents:
            a.new_adopter = 0
            a.num_infected_neighbors = sum(
                agents.get_agent(n).preference_state == 1
                for _, n in a.network.get_neighbors(a)
            )
        for source in sources:
            for _, neighbor_id in source.network.get_neighbors(source):
                target = agents.get_agent(neighbor_id)
                if target is None or target.id == source.id:
                    continue
                event_id = f'{self.run_id}:{len(self.diffusion_events)}'
                # This record is frozen BEFORE the target receives the current message.
                pre = dict(
                    event_id=event_id, pair_id=self.pair_id, run_id=self.run_id,
                    condition=self.condition, period=period,
                    source_agent=source.id, target_agent=target.id,
                    received_post=source.post, profile_before=copy.deepcopy(target.profile),
                    exposure_count_before=target.exposure_count,
                    estimated_stance_before=self.observer_states[target.id]['estimated_stance'],
                    predicted_stance_after_persistence=self.observer_states[target.id]['estimated_stance'],
                    predicted_delta_zero=0.0,
                    degree_centrality=target.degree_centrality,
                    betweenness_centrality=target.betweenness_centrality,
                    received_history_json=json.dumps(target.received_message_history, ensure_ascii=False),
                    generated_history_json=json.dumps(target.generated_message_history, ensure_ascii=False),
                    behavior_history_json=json.dumps(target.behavior_history, ensure_ascii=False),
                )
                pre.update(self.public_features(target))
                pre['estimated_stance_before'] = pre['estimated_stance']
                pre['source_estimated_stance'] = self.public_features(source)['estimated_stance']
                pre['estimated_alignment'] = 1 - abs(pre['estimated_stance'] - pre['source_estimated_stance']) / 2
                before = len(self.diffusion_events)
                target.exposure_count += 1
                fn = target.infection_stance_aware if stance_aware else target.infection
                fn(agents, period, self.diffusion_events, self.message_events, source)
                if len(self.diffusion_events) == before:
                    target.exposure_count -= 1
                    continue
                if len(self.diffusion_events) != before + 1:
                    raise AssertionError('Exactly one final event is required per exposure.')
                event = self.diffusion_events[-1]
                message = self.message_events[-1]
                metadata = dict(event_id=event_id, pair_id=self.pair_id,
                                run_id=self.run_id, condition=self.condition)
                event.update(metadata)
                message.update(metadata)
                # Unsigned embedding distances are text-change proxies, not stance estimates.
                for old, new in [('estimated_stance_shift', 'text_change_product'),
                                 ('estimated_stance_shift_post', 'post_cosine_distance'),
                                 ('estimated_stance_shift_profile', 'profile_cosine_distance')]:
                    message[new] = message.pop(old, None)
                message['parent_message_id'] = source.message_id
                message['original_message_id'] = source.original_message_id or source.message_id
                # Rejections without generation must not masquerade as new narrative nodes.
                if not event['adopted'] and not stance_aware:
                    message['message_id'] = None
                    message['generated_post'] = None
                    message['embedding_generated_post'] = None
                target.received_message_history.append(source.post)
                if message.get('generated_post'):
                    target.generated_message_history.append(message['generated_post'])
                behavior = str(event['behavior']).lower()
                target.behavior_history.append(behavior)
                target.interaction_history.append(dict(period=period, source=source.id,
                                                       message=source.post, behavior=behavior))
                if event['adopted']:
                    target.influencer = source
                    target.accept = source.post
                    target.accept_embedding = source.post_embedding
                # Observer sees public traces only, for both acceptances AND rejections.
                self.update_observer(target, period)
                event['estimated_stance_before'] = pre['estimated_stance_before']
                event['estimated_stance_after'] = self.observer_states[target.id]['estimated_stance']
                self.observable_events.append(pre)
                self.transmissions.append(dict(
                    event_id=event_id, run_id=self.run_id, period=period,
                    source_agent=source.id, target_agent=target.id,
                    message_id=source.message_id,
                    root_message_id=source.original_message_id or source.message_id,
                    public_text=source.post))
                self.prediction_labels.append(dict(
                    **metadata, period=period, target_agent=target.id,
                    target_stance_before=event['target_stance_before'],
                    target_stance_after=event['target_stance_after'],
                    target_delta_stance=event['target_delta_stance'], adopted=event['adopted'],
                ))
        self.record_trajectory(agents, period)
        self._period_index += 1
        return all(a.preference_state == 1 for a in agents)

    def save_diffusion_events(self, method, k):
        # A unique pair folder prevents overwrites between repetitions.
        folder = os.path.join('data', 'evaluation', self.pair_id, self.condition)
        os.makedirs(folder, exist_ok=True)
        tables = dict(diffusion_events=self.diffusion_events, message_events=self.message_events,
                      observable_preinteraction=self.observable_events,
                      prediction_labels_PRIVATE=self.prediction_labels,
                      agent_trajectory_PRIVATE=self.trajectories,
                      candidate_snapshots=self.candidate_snapshots,
                      transmission_events=self.transmissions)
        for name, rows in tables.items():
            pd.DataFrame(rows).to_csv(os.path.join(folder, name + '.csv'), index=False,
                                      encoding='utf-8-sig')
        with open(os.path.join(folder, 'node_histories.json'), 'w', encoding='utf-8') as out:
            json.dump({str(a.id): dict(dominant_decision_style=a.dominant_decision_style,
                       topic_exposures=a.topic_exposures, decisions=a.decision_history)
                       for a in self._history_agents}, out, ensure_ascii=False, indent=2)
        frame = pd.DataFrame(self.diffusion_events)
        metrics = dict(pair_id=self.pair_id, run_id=self.run_id, condition=self.condition,
                       interactions=len(frame), agents=len(self._initial_agents))
        if len(frame):
            delta = frame.target_delta_stance.to_numpy(dtype=float)
            metrics.update(new_adopters=int(frame['new_adoption'].sum()) if 'new_adoption' in frame else int(frame.adopted.sum()),
                           mean_absolute_shift=float(np.abs(delta).mean()),
                           zero_shift_rmse=float(np.sqrt(np.mean(delta ** 2))))
        else:
            metrics.update(new_adopters=0, mean_absolute_shift=None, zero_shift_rmse=None)
        metrics['seed_count'] = sum(s.get('isSeed', False) for s in self._initial_agents.values())
        metrics['final_adopted'] = metrics['seed_count'] + metrics['new_adopters']
        with open(os.path.join(folder, 'metrics.json'), 'w', encoding='utf-8') as out:
            json.dump(metrics, out, indent=2)
        manifest = dict(pair_id=self.pair_id, condition=self.condition,
                        seed_ids=[i for i, s in self._initial_agents.items() if s.get('isSeed')],
                        random_seed=getattr(self.scenario, 'random_seed', None),
                        observation_periods=self._period_index,
                        stance_topic=getattr(self.scenario, 'stance_topic', None),
                        text_stance_observer=getattr(self.scenario, 'text_stance_observer', False),
                        split_group='pair_id',
                        warning='Only observable_preinteraction.csv is safe for pre-interaction features. '
                                'Other tables contain outcomes and/or private simulator state. '
                                'LLM sampling is not guaranteed reproducible by Python/NumPy seeds.')
        with open(os.path.join(folder, 'manifest.json'), 'w', encoding='utf-8') as out:
            json.dump(manifest, out, indent=2)
        print('Evaluation saved:', folder)
