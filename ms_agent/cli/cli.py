import argparse
import sys
import io
import contextlib
from typing import Tuple, List

from ms_agent.cli.app import AppCMD
from ms_agent.cli.run import RunCMD
from ms_agent.cli.ui import UICMD


class TeeTextWriter(io.StringIO):
    def __init__(self, mirror):
        super().__init__()
        self._mirror = mirror

    def write(self, value: str) -> int:
        written = super().write(value)
        try:
            self._mirror.write(value)
            self._mirror.flush()
        except Exception:
            pass
        return written


def run_in_process(args: List[str]) -> Tuple[int, str, str]:
    """Run the ms-agent CLI command in-process, capturing stdout and stderr.

    Args:
        args: Command-line arguments, including the executable name as args[0].

    Returns:
        A tuple of (returncode, stdout, stderr).
    """
    stdout_buffer = io.StringIO()
    stderr_mirror = sys.__stderr__ if sys.__stderr__ is not None else sys.stderr
    stderr_buffer = TeeTextWriter(stderr_mirror)
    
    original_argv = sys.argv[:]
    returncode = 0
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            sys.argv = args[:]
            try:
                run_cmd()
            except SystemExit as exc:
                code = exc.code
                if code is None:
                    returncode = 0
                elif isinstance(code, int):
                    returncode = code
                else:
                    stderr_buffer.write(str(code))
                    returncode = 1
    except Exception as exc:
        returncode = 2
        stderr_buffer.write(f"\nfailed to run ms-agent CLI in-process: {exc}")
    finally:
        sys.argv = original_argv

    return returncode, stdout_buffer.getvalue(), stderr_buffer.getvalue()


def run_cmd():
    """This is the entrance of the all the cli commands.

    This cmd imports all other sub commands, for example, `run` and `app`.
    """
    parser = argparse.ArgumentParser(
        'ModelScope-agent Command Line tool',
        usage='ms-agent <command> [<args>]')

    subparsers = parser.add_subparsers(
        help='ModelScope-agent commands helpers')

    RunCMD.define_args(subparsers)
    AppCMD.define_args(subparsers)
    UICMD.define_args(subparsers)

    # unknown args will be handled in config.py
    args, _ = parser.parse_known_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        exit(1)
    cmd = args.func(args)
    cmd.execute()


if __name__ == '__main__':
    run_cmd()
