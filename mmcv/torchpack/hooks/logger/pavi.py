from __future__ import print_function

import os
import time
from datetime import datetime
from threading import Thread

import requests
from six.moves.queue import Empty, Queue

from .base import LoggerHook
from ...utils import master_only, get_host_info


class PaviClient(object):

    def __init__(self, url, username=None, password=None, instance_id=None):
        self.url = url
        self.username = self._get_env_var(username, 'PAVI_USERNAME')
        self.password = self._get_env_var(password, 'PAVI_PASSWORD')
        self.instance_id = instance_id
        self.log_queue = None

    def _get_env_var(self, var, env_var):
        if var is not None:
            return str(var)

        var = os.getenv(env_var)
        if not var:
            raise ValueError(
                '"{}" is neither specified nor defined as env variables'.
                format(env_var))
        return var

    def connect(self,
                model_name,
                work_dir=None,
                info=dict(),
                timeout=5,
                logger=None):
        if logger:
            log_info = logger.info
            log_error = logger.error
        else:
            log_info = log_error = print
        log_info('connecting pavi service {}...'.format(self.url))
        post_data = dict(
            time=str(datetime.now()),
            username=self.username,
            password=self.password,
            instance_id=self.instance_id,
            model=model_name,
            work_dir=os.path.abspath(work_dir) if work_dir else '',
            session_file=info.get('session_file', ''),
            session_text=info.get('session_text', ''),
            model_text=info.get('model_text', ''),
            device=get_host_info())
        try:
            response = requests.post(self.url, json=post_data, timeout=timeout)
        except Exception as ex:
            log_error('fail to connect to pavi service: {}'.format(ex))
        else:
            if response.status_code == 200:
                self.instance_id = response.text
                log_info('pavi service connected, instance_id: {}'.format(
                    self.instance_id))
                self.log_queue = Queue()
                self.log_thread = Thread(target=self.post_worker_fn)
                self.log_thread.daemon = True
                self.log_thread.start()
                return True
            else:
                log_error('fail to connect to pavi service, status code: '
                          '{}, err message: {}'.format(response.status_code,
                                                       response.reason))
        return False

    def post_worker_fn(self, max_retry=3, queue_timeout=1, req_timeout=3):
        while True:
            try:
                log = self.log_queue.get(timeout=queue_timeout)
            except Empty:
                time.sleep(1)
            except Exception as ex:
                print('fail to get logs from queue: {}'.format(ex))
            else:
                retry = 0
                while retry < max_retry:
                    try:
                        response = requests.post(
                            self.url, json=log, timeout=req_timeout)
                    except Exception as ex:
                        retry += 1
                        print('error when posting logs to pavi: {}'.format(ex))
                    else:
                        status_code = response.status_code
                        if status_code == 200:
                            break
                        else:
                            print('unexpected status code: %d, err msg: %s',
                                  status_code, response.reason)
                            retry += 1
                if retry == max_retry:
                    print('fail to send logs of iteration %d', log['iter_num'])

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
                 interval=10,
                 reset_meter=True,
                 ignore_last=True):
        self.pavi = PaviClient(url, username, password, instance_id)
        super(PaviLoggerHook, self).__init__(interval, reset_meter,
                                             ignore_last)

    @master_only
    def connect(self,
                model_name,
                work_dir=None,
                info=dict(),
                timeout=5,
                logger=None):
        return self.pavi.connect(model_name, work_dir, info, timeout, logger)

    @master_only
    def log(self, runner):
        log_outs = runner.log_buffer.output.copy()
        log_outs.pop('time', None)
        log_outs.pop('data_time', None)
        self.pavi.log(runner.mode, runner.iter, log_outs)
