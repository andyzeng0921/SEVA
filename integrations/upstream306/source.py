"""Import unmodified upstream entry points without starting foreign robot drivers."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

PROJECT = Path(os.environ.get('ZENG_PROJECT_ROOT', Path(__file__).resolve().parents[2])).resolve()
UPSTREAM = PROJECT / 'upstream'


def checked_path(relative):
    path = (UPSTREAM / relative).resolve()
    path.relative_to(UPSTREAM.resolve())
    hashes = json.loads((UPSTREAM / 'entrypoints.sha256.json').read_text())
    if hashes.get(relative) != hashlib.sha256(path.read_bytes()).hexdigest():
        raise RuntimeError('Upstream source integrity check failed: ' + relative)
    return path


def module(relative, name):
    path = checked_path(relative)
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def definitions(relative, names, namespace=None):
    """Load exact function bodies, excluding ROS1/AI2THOR import-time dependencies.

    This packaging adapter does not rewrite AST nodes or synthesize algorithms.
    The complete source file remains on disk; every executed function retains its
    original filename/line numbers. Only a hardcoded caller whitelist is accepted.
    """
    path = checked_path(relative)
    parsed = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in parsed.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {n.name for n in nodes} != set(names):
        raise RuntimeError('Upstream entry-point signature changed: ' + relative)
    result = dict(namespace or {})
    result['__file__'] = str(path)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), result)
    return result


def stretch_task():
    sys.path.insert(0, str(UPSTREAM / 'stretch_ai/src'))
    return module('stretch_ai/src/stretch/core/task.py', '_upstream306_stretch_task')


def physmem():
    sys.path.insert(0, str(UPSTREAM / 'physmem'))
    import physmem as package
    return package


def limp():
    return module('robotlimp/limp/language/temporal_logic/ltl_progression.py', '_upstream306_limp')


def bumble_prompts():
    return definitions('bumble/bumble/tiago/skills/selector.py',
                       ['make_prompt', 'make_history_prompt', 'make_cross_history_prompt'])
