import logging


class LoggingMixin:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        log = logging.getLogger(self.__class__.__name__)
        log.handlers.clear()
        log.propagate = False
        log.setLevel(logging.DEBUG)
        log.addHandler(handler)
        self.log = log
