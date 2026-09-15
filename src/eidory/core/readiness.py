"""Report capabilities independently; a reachable server is not proof of generation."""
import json
import urllib.error
import urllib.request


def check_model_service(endpoint, model='', api_key='', *, generate=False, timeout=8):
    endpoint = endpoint.strip().rstrip('/')
    if not endpoint:
        return False, '服务地址为空；在设置中填写以 /v1 结尾的 API 地址。'
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    try:
        request = urllib.request.Request(endpoint + '/models', headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            doc = json.load(response)
        models = [r['id'] for r in doc.get('data', []) if isinstance(r, dict) and isinstance(r.get('id'), str)]
        if not models:
            return False, '服务可连接，但未列出模型；请先加载一个文本模型。'
        if model and model not in models:
            return False, f'服务可连接，但选定模型 {model} 不在模型列表中；请检查名称或加载该模型。'
        chosen = model or models[0]
        if not generate:
            return True, f'{chosen} 可见；尚未验证实际生成。点击“测试 AI 生成”进一步检查。'
        body = {'model': chosen, 'messages': [{'role': 'user', 'content': '请只用一句中文描述一张桌子，直接回答，不需要分析。 /no_think'}], 'max_tokens': 1200, 'temperature': 0, 'stream': False}
        request = urllib.request.Request(endpoint + '/chat/completions', data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=max(timeout, 90)) as response:
            generated = json.load(response)
        content = generated.get('choices', [{}])[0].get('message', {}).get('content', '')
        if not isinstance(content, str) or not content.strip():
            return False, '请求成功但未返回正文；请检查模型的输出格式或增加模型输出额度。'
        return True, '实际生成成功：' + content.strip()[:250]
    except urllib.error.HTTPError as exc:
        fixes = {401: '认证失败；检查 API Key。', 403: '访问被拒绝；检查账户权限或服务访问限制。',
                 404: '接口不存在；检查 API 地址及 /v1 路径。', 429: '请求受限；等待后重试并检查服务额度。'}
        return False, f'HTTP {exc.code}：' + fixes.get(exc.code, '服务请求失败；检查模型服务日志。')
    except (ValueError, KeyError, IndexError, TypeError):
        return False, '服务返回了无法解析的内容；请确认地址指向兼容的模型 API。'
    except (OSError, urllib.error.URLError):
        return False, '连接失败或超时；启动模型服务，并检查地址、端口和网络。'


def library_readiness(store):
    with store.connect() as conn:
        total = conn.execute('SELECT COUNT(*) FROM images WHERE is_missing=0').fetchone()[0]
        missing = conn.execute('SELECT COUNT(*) FROM images WHERE is_missing=1').fetchone()[0]
        models = conn.execute("SELECT model_name, COUNT(DISTINCT e.image_id) AS amount FROM embeddings e JOIN images i ON i.id=e.image_id WHERE e.status='ready' AND i.is_missing=0 GROUP BY model_name").fetchall()
    return [('图库浏览', total > 0, f'{total} 个可用文件，{missing} 个丢失。' + ('请先导入图片。' if not total else '')),
            ('语义索引', bool(models), '；'.join(f"{r[0]}：{r[1]}/{total} 张" for r in models) + '。覆盖仅表示索引记录；选用模型须与索引一致。' if models else '尚无可用语义索引；打开后台任务与索引，为图库建立索引。')]
