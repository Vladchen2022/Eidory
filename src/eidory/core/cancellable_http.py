"""Cancellable model requests with a bounded, disposable network process.

Stopping the client cannot certify that a remote model stopped computing.
No database or UI objects ever enter the child process.
"""
import multiprocessing
import time

import requests


class RequestCancelled(requests.RequestException):
    pass


def _post(connection, url, payload, headers, timeout, method):
    try:
        request = requests.get if method == 'GET' else requests.post
        kwargs = {} if method == 'GET' else {'json': payload}
        with request(url, headers=headers, timeout=timeout, stream=True, **kwargs) as response:
            chunks, size = [], 0
            for chunk in response.iter_content(64 * 1024):
                size += len(chunk)
                if size > 8 * 1024 * 1024:
                    raise requests.RequestException('模型响应超过 8 MB')
                chunks.append(chunk)
            connection.send(('response', response.status_code, dict(response.headers), b''.join(chunks)))
    except Exception as exc:
        kind = 'timeout' if isinstance(exc, requests.Timeout) else 'connection' if isinstance(exc, requests.ConnectionError) else 'error'
        connection.send((kind, str(exc)))
    finally:
        connection.close()


def post_json(url, *, json, headers, timeout, cancel=None, method='POST'):
    if cancel is None:
        if method == 'GET':
            return requests.get(url, headers=headers, timeout=timeout)
        return requests.post(url, json=json, headers=headers, timeout=timeout)
    if cancel.is_set():
        raise RequestCancelled('已停止本地等待；服务端状态未确认')
    context = multiprocessing.get_context('spawn')
    incoming, outgoing = context.Pipe(duplex=False)
    process = context.Process(target=_post, args=(outgoing, url, json, headers, timeout, method), daemon=True)
    deadline = time.monotonic() + timeout
    try:
        process.start()
        outgoing.close()
        while not incoming.poll(.05):
            if cancel.is_set():
                raise RequestCancelled('已停止本地等待；服务端状态未确认')
            if time.monotonic() > deadline:
                raise requests.Timeout(f'模型请求在 {timeout} 秒内未完成')
            if not process.is_alive():
                raise requests.ConnectionError('模型请求进程意外结束')
        if cancel.is_set():
            raise RequestCancelled('已停止本地等待；服务端状态未确认')
        try:
            result = incoming.recv()
        except EOFError as exc:
            raise requests.ConnectionError('模型请求进程意外结束') from exc
        if result[0] != 'response':
            error = {'timeout': requests.Timeout, 'connection': requests.ConnectionError}.get(result[0], requests.RequestException)
            raise error(result[1])
        response = requests.Response()
        response.status_code, response.headers, response._content = result[1:]
        response.url = url
        response.encoding = 'utf-8'
        return response
    finally:
        incoming.close()
        outgoing.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(.5)
            if process.is_alive():
                process.kill()
                process.join(.5)
            process.close()


def get_json(url, *, headers, timeout, cancel=None):
    return post_json(url, json=None, headers=headers, timeout=timeout, cancel=cancel, method='GET')
