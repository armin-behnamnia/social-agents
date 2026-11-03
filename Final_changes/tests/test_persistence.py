import json
from pathlib import Path
import tempfile
import types
import unittest

from source.persistence import Audit, AuditedClient, _ACTIVE, audit_run


class PersistenceTests(unittest.TestCase):
    def test_requests_responses_and_errors_are_durable(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = Audit(types.SimpleNamespace(audit_output_root=tmp, initial_message_type='covid'))
            token = _ACTIVE.set(audit)
            try:
                def create(**kw):
                    if kw['messages'] == 'bad':
                        raise ConnectionError('offline')
                    return {'choices': [{'message': {'content': 'ACCEPT'}}]}
                client = AuditedClient(types.SimpleNamespace(chat=types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=create))))
                client.chat.completions.create(messages='hello')
                with self.assertRaises(ConnectionError):
                    client.chat.completions.create(messages='bad')
                rows = [json.loads(s) for s in (audit.folder / 'journal_PRIVATE.jsonl').read_text().splitlines()]
                self.assertEqual([r['kind'] for r in rows],
                                 ['llm_request', 'llm_response', 'llm_request', 'llm_error'])
            finally:
                _ACTIVE.reset(token)

    def test_failed_run_retains_partial_state_and_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = types.SimpleNamespace(_agents_for_export=[], message_events=[{'text': 'preserved'}])
            model = types.SimpleNamespace(scenario=types.SimpleNamespace(audit_output_root=tmp), environment=env)
            @audit_run
            def run(model):
                raise ConnectionError('simulated API failure')
            with self.assertRaises(ConnectionError):
                run(model)
            folder = next(Path(tmp).iterdir())
            self.assertEqual(json.loads((folder / 'status.json').read_text())['status'], 'failed')
            snapshot = json.loads((folder / 'checkpoint_PRIVATE.json').read_text())
            self.assertEqual(snapshot['tables']['message_events'], [{'text': 'preserved'}])
            self.assertEqual(snapshot['phase'], 'failed_partial')

if __name__ == '__main__':
    unittest.main()
