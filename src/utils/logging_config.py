import logging.config
import os 

import sys
ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import LOG_DIR

# Create the directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)


class CustomFormatter(logging.Formatter):
    def format(self, record):
        """
        Format the specified record as text.

        Overrides logging.Formatter's format method to add log format
        to new lines in the log message.

        Args:
            record (LogRecord): record of the log.

        Returns:
            str: Formatted log record as a string with added log format for new lines.
        """
        original_msg = record.msg
        s = super().format(record)
        lines = s.split('\n')
        for i in range(1, len(lines)):
            record.msg = lines[i]
            lines[i] = super().format(record)
        record.msg = original_msg  # Reset the original message
        return '\n'.join(lines)

    

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            '()': CustomFormatter,
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',  # Default is stderr
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': os.path.join(LOG_DIR, 'default.log'),  
            'mode': 'a',
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)


TEST_LOGGER = False
if TEST_LOGGER is True:
    # Get a logger and use it
    logger = logging.getLogger(__name__)  # gets the name of the current module
    logger.info("This is an info message")
