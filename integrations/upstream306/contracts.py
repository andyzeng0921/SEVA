"""Translate the existing skill catalog to the executor's canonical wire schema."""
import math
from jsonschema import Draft202012Validator

READ_ONLY = frozenset({'robot_status', 'list_waypoints', 'observe_rgbd'})
ALIASES = {'navigate_to_waypoint': {'waypoint': 'name'}, 'save_waypoint': {'waypoint': 'name'},
           'arm_preset': {'preset': 'action_name'}, 'arm_continuous_move': {'distance_m': 'amount_meters'}}


def schema_for(package):
    aliases = ALIASES.get(package.intent, {})
    properties, required = {}, []
    for key, description in package.parameters.items():
        key = aliases.get(key, key)
        item = {k: v for k, v in description.items() if k not in {'required', 'allowed'}}
        if item['type'] == 'enum':
            item['type'], item['enum'] = 'string', description['allowed']
        properties[key] = item
        if description.get('required'):
            required.append(key)
    if package.intent in {'navigate_to_waypoint', 'save_waypoint'}:
        properties['name'].update(minLength=1, maxLength=64, pattern=r'^[\w-]+$')
    if package.intent == 'arm_continuous_move':
        required = ['direction', 'amount_meters']
        properties['amount_meters']['exclusiveMinimum'] = 0
        properties['limb'] = {'type': 'string', 'enum': ['left', 'right'], 'default': 'left'}
    if package.intent == 'arm_cross_wave':
        properties['phase'] = {'type': 'string', 'enum': ['left_up', 'right_up', 'full_cycle'], 'default': 'full_cycle'}
    if package.intent == 'vision_navigate':
        properties['goal_distance_m'].update(default=2.0, exclusiveMinimum=0)
    if package.intent == 'vision_turn':
        properties['direction']['enum'] = ['left', 'right']
        properties['direction']['default'] = 'left'
        properties['angle_deg']['default'] = 30
    if package.intent == 'fire_search':
        properties['instruction'] = {'type': 'string', 'minLength': 1, 'maxLength': 6000,
                                     'default': '搜索并靠近消防栓'}
    return {'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False}


def normalize(package, slots):
    if not isinstance(slots, dict):
        raise ValueError('slots must be an object')
    canonical = dict(slots)
    for alias, key in ALIASES.get(package.intent, {}).items():
        if alias in canonical:
            if key in canonical and canonical[key] != canonical[alias]:
                raise ValueError(f'Conflicting parameters: {alias} / {key}')
            canonical[key] = canonical.pop(alias)
    schema = schema_for(package)
    for key, item in schema['properties'].items():
        if key not in canonical and 'default' in item:
            canonical[key] = item['default']
    if any(isinstance(v, float) and not math.isfinite(v) for v in canonical.values()):
        raise ValueError('Non-finite numeric parameter')
    errors = sorted(Draft202012Validator(schema).iter_errors(canonical), key=lambda e: str(e.path))
    if errors:
        raise ValueError(errors[0].message)
    return canonical
