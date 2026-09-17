"""306 evidence -> original REFLECT summary and failure-analysis entry points."""
import json
import os
from pathlib import Path
import pickle
import shutil
import threading
import hashlib
import subprocess
import sys
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import requests

from .source import UPSTREAM, definitions, checked_path

_cwd_lock = threading.RLock()


def original_functions():
    context = {'os': os, 'json': json, 'pickle': pickle, 'np': np, 'audio2label': {}}
    utils = definitions('reflect/main/utils.py',
        ['get_robot_plan', 'convert_step_to_timestep', 'convert_timestep_to_step'], context)
    context.update(utils)
    return definitions('reflect/main/exp.py',
        ['get_scene_text', 'get_held_object', 'generate_summary', 'run_reasoning'], context)


class ConfiguredPrompter:
    """Transport adapter for the old REFLECT query protocol; bounded SDK calls."""
    def __init__(self, config):
        self.config = config

    def query(self, prompt, sampling_params, save, save_dir):
        if not self.config.api_key:
            raise RuntimeError('Configured model credential missing')
        payload = {'model': self.config.model,
                  'messages': [{'role': 'system', 'content': prompt['system']},
                               {'role': 'user', 'content': prompt['user']}],
                  'temperature': sampling_params.get('temperature', 0),
                  'max_tokens': min(sampling_params.get('max_tokens', 1024), 2048)}
        if self.config.enable_thinking is not None:
            payload['chat_template_kwargs'] = {'enable_thinking': self.config.enable_thinking}
        response = requests.post(self.config.base_url.rstrip('/') + '/chat/completions',
            headers={'Authorization': 'Bearer ' + self.config.api_key}, json=payload, timeout=45)
        if not response.ok:
            raise RuntimeError(f'Configured model returned HTTP {response.status_code}')
        choice = response.json()['choices'][0]
        if choice.get('finish_reason') == 'length' or not choice['message'].get('content'):
            raise RuntimeError('Configured model returned an incomplete conclusion')
        return choice['message']['content'].strip(), None


def summarize(task, runtime, llm_config=None, prompter=None):
    if prompter is not None:
        # Transport injection is only for deterministic offline unit tests.
        return _summarize_in_process(task, runtime, prompter=prompter)
    payload = {'task': task, 'runtime': str(Path(runtime).resolve()),
               'llm_config': asdict(llm_config) if llm_config is not None else None}
    result = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'upstream306.reflect_worker'],
        input=json.dumps(payload, ensure_ascii=False), text=True, encoding='utf-8',
        capture_output=True, timeout=155 if llm_config is not None else 20)
    if result.returncode:
        raise RuntimeError('REFLECT worker failed; no verified conclusion available')
    return json.loads(result.stdout)


def _summarize_in_process(task, runtime, llm_config=None, prompter=None):
    # REFLECT's file API assumes relative paths. A dedicated job tree preserves
    # that API and prevents input/output writes inside the unmodified repository.
    evidence = json.dumps({'task_id': task['task_id'], 'text': task['text'], 'steps': task['steps']},
                          sort_keys=True, ensure_ascii=False).encode()
    digest = hashlib.sha256(evidence).hexdigest()
    # Upstream skips existing summary files; content-based directories prevent
    # a previous partial task snapshot from becoming a stale final summary.
    job = Path(runtime).resolve() / 'reflect' / (digest + ('-reasoning' if llm_config or prompter else '-summary'))
    work = job / 'main'
    folder = 'episode'
    state = work / 'state_summary' / folder
    graphs = state / 'local_graphs'
    source = work / 'thor_tasks' / folder
    for p in [graphs, source, job / 'LLM']:
        p.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checked_path('reflect/LLM/prompts.json'), job / 'LLM/prompts.json')
    steps = [s for s in task['steps'] if s['status'] != 'pending']
    if not steps:
        return {'status': 'no_execution_evidence', 'task_id': task['task_id']}
    metadata = {'name': task['text'], 'sounds': {},
                'success_condition': 'Read operations have valid responses; physical actions require independently verified effects',
                'gt_failure_reason': None, 'gt_failure_step': None}
    (source / 'task.json').write_text(json.dumps(metadata, ensure_ascii=False))
    (state / 'L1_key_frames.txt').write_text(''.join(f'{i+1}\n' for i in range(len(steps))))
    interactions = {}
    last_graph = None
    for i, step in enumerate(steps):
        interactions[i+1] = step['intent']
        # These are explicitly execution-state observations, not invented scene
        # objects. The shape implements REFLECT's node-name/edge read protocol.
        fact = ('Execution evidence ' + json.dumps({k: step.get(k) for k in
                ['intent', 'status', 'controller_status', 'controller_success', 'effect_status',
                 'failure_code', 'verification', 'message', 'evidence_id', 'mode']},
                ensure_ascii=False)).replace('\n', ' ')
        # Upstream get_scene_text uses set(nodes). Plain strings are hashable;
        # this small record class supplies its requested get_name interface.
        last_graph = EvidenceGraph([EvidenceNode(fact)])
        with (graphs / f'local_sg_{i}.pkl').open('wb') as f:
            pickle.dump(last_graph, f)
    functions = original_functions()
    with _cwd_lock:
        old_cwd = os.getcwd()
        try:
            os.chdir(work)
            functions['generate_summary'](folder, list(range(len(steps))), {}, interactions, 0, [])
            if llm_config is not None or prompter is not None:
                functions['run_reasoning'](folder, prompter or ConfiguredPrompter(llm_config), last_graph)
        finally:
            os.chdir(old_cwd)
    output = {'source': 'REFLECT original generate_summary / run_reasoning',
              'evidence_type': '306 execution state (no AI2THOR ground truth or fabricated vision)',
              'L1': (state / 'state_summary_L1.txt').read_text(),
              'L2': (state / 'state_summary_L2.txt').read_text(), 'directory': str(state)}
    if (state / 'reasoning.json').exists():
        output['reasoning'] = json.loads((state / 'reasoning.json').read_text())
    return output


class EvidenceNode:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


class EvidenceGraph:
    def __init__(self, nodes):
        self.nodes, self.edges = nodes, {}
