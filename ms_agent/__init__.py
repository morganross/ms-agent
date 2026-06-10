# Copyright (c) ModelScope Contributors. All rights reserved.
import logging
import os
from typing import Any

if os.getenv('MS_AGENT_NO_FILE_LOG') == '1':
    _ORIGINAL_FILE_HANDLER = logging.FileHandler

    def _is_msagent_default_log(filename: Any) -> bool:
        try:
            path = os.fspath(filename)
        except TypeError:
            return False
        return os.path.basename(str(path)) == "ms_agent.log"

    class AcmNoFileHandler(_ORIGINAL_FILE_HANDLER):
        def __init__(
            self,
            filename: Any,
            mode: str = "a",
            encoding: str | None = None,
            delay: bool = False,
            errors: str | None = None,
        ) -> None:
            if not _is_msagent_default_log(filename):
                super().__init__(filename, mode=mode, encoding=encoding, delay=delay, errors=errors)
                self._acm_no_file_sink = False
                return

            logging.Handler.__init__(self)
            self.baseFilename = os.path.abspath(os.fspath(filename))
            self.mode = mode
            self.encoding = encoding
            self.errors = errors
            self.delay = delay
            self.stream = None
            self._acm_no_file_sink = True

        def emit(self, record: logging.LogRecord) -> None:
            if getattr(self, "_acm_no_file_sink", False):
                return
            super().emit(record)

        def close(self) -> None:
            if not getattr(self, "_acm_no_file_sink", False):
                super().close()
                return
            self.acquire()
            try:
                logging.Handler.close(self)
            finally:
                self.release()

    logging.FileHandler = AcmNoFileHandler

from .agent.llm_agent import LLMAgent
from .cli.cli import run_in_process
