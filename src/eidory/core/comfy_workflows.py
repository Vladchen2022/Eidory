"""Built-in reference workflows, with an intact legacy binding and custom slot."""
from __future__ import annotations

import json

from eidory.core.comfy_reference import prepare_template


TURBO_NAME = 'Viggle Turbo · 6 步'
ORIGINAL_NAME = '千问原版 · 40 步'


def official_turbo_graph():
    """Official Qwen 2.1 Turbo schedule used by the local ComfyUI Launcher."""
    graph = builtin_template()['graph']
    graph['1']['inputs']['unet_name'] = 'qwen_image_2.1_turbo_bf16.safetensors'
    del graph['11']
    graph['12']['inputs']['model'] = ['1', 0]
    graph['15'] = {'class_type': 'ManualSigmas', 'inputs': {
        'sigmas': '1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0'}}
    return graph


def builtin_template(turbo=True):
    graph = {
        '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'qwen_image_2.1_bf16.safetensors', 'weight_dtype': 'default'}},
        '2': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'qwen3vl_8b_bf16.safetensors', 'type': 'qwen_image', 'device': 'default'}},
        '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'qwen_image_2.1_vae_bf16.safetensors'}},
        '4': {'class_type': 'TextEncodeQwenImage21', 'inputs': {'clip': ['2', 0], 'prompt': '', 'negative_prompt': '', 'resolution': 1024}},
        '5': {'class_type': 'EmptyLatentImage', 'inputs': {'width': 1024 if turbo else 1152, 'height': 1024 if turbo else 2048, 'batch_size': 1}},
        '7': {'class_type': 'VAEDecode', 'inputs': {'samples': ['6', 0], 'vae': ['3', 0]}},
        '8': {'class_type': 'PreviewImage', 'inputs': {'images': ['7', 0]}},
    }
    if turbo:
        graph.update({
            '6': {'class_type': 'SamplerCustomAdvanced', 'inputs': {'noise': ['13', 0], 'guider': ['12', 0], 'sampler': ['14', 0], 'sigmas': ['15', 0], 'latent_image': ['5', 0]}},
            '11': {'class_type': 'ViggleTurboLora', 'inputs': {'model': ['1', 0], 'lora_name': 'Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors', 'strength': 1.0}},
            '12': {'class_type': 'BasicGuider', 'inputs': {'model': ['11', 0], 'conditioning': ['4', 0]}},
            '13': {'class_type': 'RandomNoise', 'inputs': {'noise_seed': 0}},
            '14': {'class_type': 'KSamplerSelect', 'inputs': {'sampler_name': 'euler'}},
            '15': {'class_type': 'ViggleTurboSigmas', 'inputs': {'latent': ['5', 0], 'nodes': '1.0, 0.9375, 0.875, 0.75, 0.5, 0.25'}},
        })
    else:
        graph['6'] = {'class_type': 'KSampler', 'inputs': {'model': ['1', 0], 'positive': ['4', 0], 'negative': ['4', 1], 'latent_image': ['5', 0], 'seed': 0, 'steps': 40, 'cfg': 1.0, 'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1.0}}
    return prepare_template(graph)


class ReferenceWorkflows:
    def __init__(self, store):
        self.store = store

    def _saved(self, key):
        value = self.store.get_setting(key, '')
        if not value:
            return None
        # Rebind older saved templates to include the explicit seed field.
        return prepare_template(json.loads(value)['graph'])

    def options(self):
        original_name = ORIGINAL_NAME
        try:
            original = self._saved('comfy.template')
            if original:
                sampler = original['graph'][original['sampler_node']]
                if sampler['class_type'] != 'KSampler' or sampler['inputs'].get('steps') != 40:
                    original_name = '原工作流 · ' + self.store.get_setting('comfy.workflow_name', '已保存')
        except (ValueError, KeyError, TypeError):
            original_name = '原工作流（需重新导入）'
        options = [('turbo', TURBO_NAME), ('original', original_name)]
        if self.store.get_setting('comfy.custom_template', ''):
            options.append(('custom', '自定义 · ' + self.store.get_setting('comfy.custom_workflow_name', 'API 工作流')))
        return options

    def selected(self):
        value = self.store.get_setting('comfy.profile', 'turbo')
        return value if value in dict(self.options()) else 'turbo'

    def template(self, profile=None):
        profile = profile or self.selected()
        if profile == 'turbo':
            template = builtin_template()
        elif profile == 'original':
            template = self._saved('comfy.template') or builtin_template(False)
        elif profile == 'custom':
            template = self._saved('comfy.custom_template')
        else:
            raise ValueError('请选择有效的生成工作流。')
        if not template:
            raise ValueError('工作流不可用，请重新导入 API 工作流。')
        return dict(template, workflow_id=profile, workflow_name=dict(self.options())[profile])

    def select(self, profile):
        self.template(profile)  # Validate before remembering the choice.
        self.store.set_setting('comfy.profile', profile)

    def import_custom(self, graph, name):
        template = prepare_template(graph)
        self.store.set_setting('comfy.custom_template', json.dumps(template, ensure_ascii=False))
        self.store.set_setting('comfy.custom_workflow_name', name)
        self.select('custom')
