import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
import numpy as np
from source.persistence import Audit, properties, serial


class NodeExportTests(unittest.TestCase):
    def test_complete_nodes_messages_and_run_labels(self):
        class Node:
            static_setting = 'inherited'
            def __init__(self, id):
                self.id = id
                self.profile = 'پروفایل آزمایشی'
                self.embedding = np.array([0.25, 0.75])
                self.received_message_history = [{'text': 'متن کامل پیام'}]
                self.decision_history = [{'decision': 'REJECT', 'reason': 'دلیل'}]
                self.custom_internal_state = {'hidden': 0.2}
                self.influencer = None
                self.network = NS(get_neighbors=lambda a: [(0, 1 - a.id)])
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [Node(0), Node(1)]
            scenario = NS(audit_output_root=tmp, initial_message_type='covid', scenario_type='Generative')
            audit = Audit(scenario)
            env = NS(_agents_for_export=nodes, _initial_node_records={a.id: properties(a) for a in nodes},
                     observer_states={0: {'estimated_stance': 0.1}}, condition='reference',
                     message_events=[dict(source_agent=0, target_agent=1, generated_post='پیام')])
            audit.append('delivery_started', source_agent=0, target_agent=1,
                         source_state=properties(nodes[0]), target_state=properties(nodes[1]))
            audit.export(env, 'completed')
            self.assertIn('covid__Generative', audit.folder.name)
            folder = audit.folder / 'exports_PRIVATE/reference'
            index = json.loads((folder / 'node_index.json').read_text())
            node = json.loads((folder / index[0]['file']).read_text())
            self.assertEqual(node['state']['embedding'], [0.25, 0.75])
            self.assertEqual(node['state']['static_setting'], 'inherited')
            self.assertEqual(node['initial_state']['custom_internal_state']['hidden'], 0.2)
            self.assertEqual(len(node['outgoing_neighbors']), 1)
            self.assertEqual(node['events']['message_events'][0]['generated_post'], 'پیام')
            self.assertTrue((audit.folder / node['timeline_file']).exists())
            rows = json.loads((folder / 'message_events.json').read_text())
            self.assertEqual(rows[0]['scenario_type'], 'Generative')
            self.assertEqual(rows[0]['initial_message_type'], 'covid')
            self.assertIn('scenario_type', (folder / 'message_events.csv').read_text())

    def test_numpy_and_sensitive_values(self):
        value = serial({'embedding': np.array([1., 2.]), 'api_key': 'secret', 'num_tokens': 50})
        self.assertEqual(value['api_key'], '[REDACTED]')
        self.assertEqual(value['num_tokens'], 50)
        self.assertEqual(value['embedding'], [1., 2.])


if __name__ == '__main__':
    unittest.main()
