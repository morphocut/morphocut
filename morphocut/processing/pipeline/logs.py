from rq import get_current_job

from morphocut.processing.pipeline import LogBase


class ObjectCountLog(LogBase):
    """Writes the progress into the current Job. The progress is accessible via job.meta.get('progress', 0).

    Note
    ----
    Do not include the `self` parameter in the ``Parameters`` section.

    Parameters
    ----------
    msg : str
        Human readable string describing the exception.
    code : :obj:`int`, optional
        Numeric error code.

    Attributes
    ----------
    msg : str
        Human readable string describing the exception.
    code : int
        Numeric error code.

    """

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.log_data[self.name] = 0

    def log(self):
        self.log_data[self.name] += 1


class ParamsLog(LogBase):
    """Writes the progress into the current Job. The progress is accessible via job.meta.get('progress', 0).

    Note
    ----
    Do not include the `self` parameter in the ``Parameters`` section.

    Parameters
    ----------
    msg : str
        Human readable string describing the exception.
    code : :obj:`int`, optional
        Numeric error code.

    Attributes
    ----------
    msg : str
        Human readable string describing the exception.
    code : int
        Numeric error code.

    """

    def __init__(self, name, params):
        super().__init__()
        self.name = name
        self.log_data[self.name] = params

    def log(self):
        pass
