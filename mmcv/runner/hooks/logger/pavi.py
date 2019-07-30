from __future__ import print_function
import logging
import os
import os.path as osp
import time
from datetime import datetime
from threading import Thread

import requests
from six.moves.queue import Empty, Queue

from ...utils import get_host_info, master_only
from .base import LoggerHook


class PaviClient(object):

    def __init__(self, url, username=None, password=None, instance_id=None):
        self.url = url
        self.username = self._get_env_var(username, 'PAVI_USERNAME')
        self.password = self._get_env_var(password, 'PAVI_PASSWORD')
        self.instance_id = instance_id
        self.log_queue = None
        self.logger = None

    def _get_env_var(self, var, env_var):
        if var is not None:
            return str(var)

        var = os.getenv(env_var)
        if not var:
            raise ValueError(
                '"{}" is neither specified nor defined as env variables'.
                format(env_var))
        return var

    def _print_log(self, msg, level=logging.INFO, *args, **kwargs):
        if self.logger is not None:
            self.logger.log(level, msg, *args, **kwargs)
        else:
            print(msg, *args, **kwargs)

    def connect(self,
                model_name,
                work_dir=None,
                info=dict(),
                timeout=5,
                logger=None):
        if logger is not None:
            self.logger = logger
        self._print_log('connecting pavi service {}...'.format(self.url))
        post_data = dict(
            time=str(datetime.now()),
            username=self.username,
            password=self.password,
            instance_id=self.instance_id,
            model=model_name,
            work_dir=osp.abspath(work_dir) if work_dir else '',
            session_file=info.get('session_file', ''),
            session_text=info.get('session_text', ''),
            model_text=info.get('model_text', ''),
            device=get_host_info())
        try:
            response = requests.post(self.url, json=post_data, timeout=timeout)
        except Exception as ex:
            self._print_log(
                'fail to connect to pavi service: {}'.format(ex),
                level=logging.ERROR)
        else:
            if response.status_code == 200:
                self.instance_id = response.text
                self._print_log(
                    'pavi service connected, instance_id: {}'.format(
                        self.instance_id))
                self.log_queue = Queue()
                self.log_thread = Thread(target=self.post_worker_fn)
                self.log_thread.daemon = True
                self.log_thread.start()
                return True
            else:
                self._print_log(
                    'fail to connect to pavi service, status code: '
                    '{}, err message: {}'.format(response.status_code,
                                                 response.reason),
                    level=logging.ERROR)
        return False

    def post_worker_fn(self, max_retry=3, queue_timeout=1, req_timeout=3):
        while True:
            try:
                log = self.log_queue.get(timeout=queue_timeout)
            except Empty:
                time.sleep(1)
            except Exception as ex:
                self._print_log(
                    'fail to get logs from queue: {}'.format(ex),
                    level=logging.ERROR)
            else:
                retry = 0
                while retry < max_retry:
                    try:
                        response = requests.post(
                            self.url, json=log, timeout=req_timeout)
                    except Exception as ex:
                        retry += 1
                        self._print_log(
                            'error when posting logs to pavi: {}'.format(ex),
                            level=logging.ERROR)
                    else:
                        status_code = response.status_code
                        if status_code == 200:
                            break
                        else:
                            self._print_log(
                                'unexpected status code: {}, err msg: {}'.
                                format(status_code, response.reason),
                                level=logging.ERROR)
                            retry += 1
                if retry == max_retry:
                    self._print_log(
                        'fail to send logs of iteration {}'.format(
                            log['iter_num']),
                        level=logging.ERROR)

    def log(self, phase, iter, outputs):
        if self.log_queue is not None:
            logs = {
                'time': str(datetime.now()),
                'instance_id': self.instance_id,
                'flow_id': phase,
                'iter_num': iter,
                'outputs': outputs,
                'msg': ''
            }
            self.log_queue.put(logs)


class PaviLoggerHook(LoggerHook):

    def __init__(self,
                 url,
                 username=None,
                 password=None,
                 instance_id=None,
                 config_file=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True):
        self.pavi = PaviClient(url, username, password, instance_id)
        self.config_file = config_file
        super(PaviLoggerHook, self).__init__(interval, ignore_last, reset_flag)

    def before_run(self, runner):
        super(PaviLoggerHook, self).before_run(runner)
        self.connect(runner)

    @master_only
    def connect(self, runner, timeout=5):
        cfg_info = dict()
        if self.config_file is not None:
            with open(self.config_file, 'r') as f:
                config_text = f.read()
            cfg_info.update(
                session_file=self.config_file, session_text=config_text)
        return self.pavi.connect(runner.model_name, runner.work_dir, cfg_info,
                                 timeout, runner.logger)

    @master_only
    def log(self, runner):
        log_outs = runner.log_buffer.output.copy()
        log_outs.pop('time', None)
        log_outs.pop('data_time', None)
        for k, v in log_outs.items():
            if isinstance(v, str):
                log_outs.pop(k)
        self.pavi.log(runner.mode, runner.iter + 1, log_outs)
