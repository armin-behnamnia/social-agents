"""Offline checks; run: python tests/test_topic_integration.py from project root."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('topic_data', ROOT / 'source/topic_data.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TopicTests(unittest.TestCase):
    def scenario(self, topic, strategy='Generic', path=None):
        s = SimpleNamespace(initial_message_type=topic, scenario_type=strategy,
                            topics_excel_path=path or ROOT / 'data/topics.xlsx')
        module.apply_topic(s)
        return s

    def test_topics_and_strategies(self):
        for topic in ('covid', 'localbiz'):
            for strategy in ('Generic', 'Selective', 'Generative'):
                s = self.scenario(topic, strategy)
                self.assertTrue(s.general_influence_msg)
                self.assertTrue(s.generative_goal)
                self.assertGreaterEqual(len(s.sub_influence_msgs), 3)
                self.assertTrue(s.scenario_name.endswith('_' + strategy))

    def test_unknown(self):
        with self.assertRaises(ValueError):
            self.scenario('not-present')

    def test_new_topic_and_invalid_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'topics.xlsx'
            wb = load_workbook(ROOT / 'data/topics.xlsx')
            sheet = wb['topics']
            row = list(next(sheet.iter_rows(min_row=2, max_row=2, values_only=True)))
            headers = [c.value for c in sheet[1]]
            row[headers.index('topic')] = 'newtopic'
            sheet.append(row)
            wb.save(path)
            self.assertTrue(self.scenario('newtopic', path=path).generative_goal)
            sheet.cell(sheet.max_row, headers.index('generative_goal') + 1).value = None
            wb.save(path)
            wb.close()
            with self.assertRaises(ValueError):
                self.scenario('newtopic', path=path)

    def test_generative_prompt_and_call(self):
        tree = ast.parse((ROOT / 'source/LLM_module.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'goal_promotion')
        calls = []
        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='generated'))])
        namespace = dict(client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())), model_used='offline')
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<test>', 'exec'), namespace)
        s = self.scenario('covid', 'Generative')
        self.assertEqual(namespace['goal_promotion']('test profile', s), 'generated')
        prompt = calls[0]['messages'][0]['content']
        self.assertIn(s.generative_goal, prompt)
        self.assertNotIn('LocalBiz', prompt)
        env = (ROOT / 'source/environment.py').read_text()
        self.assertIn('goal_promotion(target_agent.profile, self.scenario)', env)


if __name__ == '__main__':
    unittest.main()
