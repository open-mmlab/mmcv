import os
import os.path as osp
import subprocess
import tempfile

from mmcv.utils import requires_executable


@requires_executable('ffmpeg')
def convert_video(in_file, out_file, print_cmd=False, pre_options='',
                  **kwargs):
    """Convert a video with ffmpeg.

    This provides a general api to ffmpeg, the executed command is::

        `ffmpeg -y <pre_options> -i <in_file> <options> <out_file>`

    Options(kwargs) are mapped to ffmpeg commands with the following rules:

    - key=val: "-key val"
    - key=True: "-key"
    - key=False: ""

    Args:
        in_file (str): Input video filename.
        out_file (str): Output video filename.
        pre_options (str): Options appears before "-i <in_file>".
        print_cmd (bool): Whether to print the final ffmpeg command.
    """
    options = []
    for k, v in kwargs.items():
        if isinstance(v, bool):
            if v:
                options.append('-{}'.format(k))
        elif k == 'log_level':
            assert v in [
                'quiet', 'panic', 'fatal', 'error', 'warning', 'info',
                'verbose', 'debug', 'trace'
            ]
            options.append('-loglevel {}'.format(v))
        else:
            options.append('-{} {}'.format(k, v))
    cmd = 'ffmpeg -y {} -i {} {} {}'.format(pre_options, in_file,
                                            ' '.join(options), out_file)
    if print_cmd:
        print(cmd)
    subprocess.call(cmd, shell=True)


@requires_executable('ffmpeg')
def resize_video(in_file,
                 out_file,
                 size=None,
                 ratio=None,
                 keep_ar=False,
                 log_level='info',
                 print_cmd=False,
                 **kwargs):
    """Resize a video.

    Args:
        in_file (str): Input video filename.
        out_file (str): Output video filename.
        size (tuple): Expected size (w, h), eg, (320, 240) or (320, -1).
        ratio (tuple or float): Expected resize ratio, (2, 0.5) means
            (w*2, h*0.5).
        keep_ar (bool): Whether to keep original aspect ratio.
        log_level (str): Logging level of ffmpeg.
        print_cmd (bool): Whether to print the final ffmpeg command.
    """
    if size is None and ratio is None:
        raise ValueError('expected size or ratio must be specified')
    elif size is not None and ratio is not None:
        raise ValueError('size and ratio cannot be specified at the same time')
    options = {'log_level': log_level}
    if size:
        if not keep_ar:
            options['vf'] = 'scale={}:{}'.format(size[0], size[1])
        else:
            options['vf'] = ('scale=w={}:h={}:force_original_aspect_ratio'
                             '=decrease'.format(size[0], size[1]))
    else:
        if not isinstance(ratio, tuple):
            ratio = (ratio, ratio)
        options['vf'] = 'scale="trunc(iw*{}):trunc(ih*{})"'.format(
            ratio[0], ratio[1])
    convert_video(in_file, out_file, print_cmd, **options)


@requires_executable('ffmpeg')
def cut_video(in_file,
              out_file,
              start=None,
              end=None,
              vcodec=None,
              acodec=None,
              log_level='info',
              print_cmd=False,
              **kwargs):
    """Cut a clip from a video.

    Args:
        in_file (str): Input video filename.
        out_file (str): Output video filename.
        start (None or float): Start time (in seconds).
        end (None or float): End time (in seconds).
        vcodec (None or str): Output video codec, None for unchanged.
        acodec (None or str): Output audio codec, None for unchanged.
        log_level (str): Logging level of ffmpeg.
        print_cmd (bool): Whether to print the final ffmpeg command.
    """
    options = {'log_level': log_level}
    if vcodec is None:
        options['vcodec'] = 'copy'
    if acodec is None:
        options['acodec'] = 'copy'
    if start:
        options['ss'] = start
    else:
        start = 0
    if end:
        options['t'] = end - start
    convert_video(in_file, out_file, print_cmd, **options)


@requires_executable('ffmpeg')
def concat_video(video_list,
                 out_file,
                 vcodec=None,
                 acodec=None,
                 log_level='info',
                 print_cmd=False,
                 **kwargs):
    """Concatenate multiple videos into a single one.

    Args:
        video_list (list): A list of video filenames
        out_file (str): Output video filename
        vcodec (None or str): Output video codec, None for unchanged
        acodec (None or str): Output audio codec, None for unchanged
        log_level (str): Logging level of ffmpeg.
        print_cmd (bool): Whether to print the final ffmpeg command.
    """
    _, tmp_filename = tempfile.mkstemp(suffix='.txt', text=True)
    with open(tmp_filename, 'w') as f:
        for filename in video_list:
            f.write('file {}\n'.format(osp.abspath(filename)))
    options = {'log_level': log_level}
    if vcodec is None:
        options['vcodec'] = 'copy'
    if acodec is None:
        options['acodec'] = 'copy'
    convert_video(
        tmp_filename,
        out_file,
        print_cmd,
        pre_options='-f concat -safe 0',
        **options)
    os.remove(tmp_filename)
