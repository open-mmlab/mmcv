import os
import subprocess


def digit_version(version_str):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions.

    Args:
        version_str (str): The version string.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return tuple(digit_version)


def _minimal_ext_cmd(cmd):
    # construct minimal environment
    env = {}
    for k in ['SYSTEMROOT', 'PATH', 'HOME']:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    # LANGUAGE is used on win32
    env['LANGUAGE'] = 'C'
    env['LANG'] = 'C'
    env['LC_ALL'] = 'C'
    out = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
    return out


def get_git_hash(fallback='unknown', digits=None):
    """Get the git hash of the current repo.

    Args:
        fallback (str, optional): The fallback string when git hash is
            unavailable. Defaults to 'unknown'.
        digits (int, optional): kept digits of the hash. Defaults to None,
            meaning all digits are kept.

    Returns:
        str: Git commit hash.
    """

    if digits is not None and not isinstance(digits, int):
        raise TypeError('digits must be None or an integer')

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
        if digits is not None:
            sha = sha[:digits]
    except OSError:
        sha = fallback

    return sha
