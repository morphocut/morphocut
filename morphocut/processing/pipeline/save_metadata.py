import json
from rq import get_current_job

from morphocut.processing.pipeline import NodeBase
from morphocut.server import models, helpers, tasks
from morphocut.server.extensions import database, migrate, redis_store
from sqlalchemy.orm.attributes import flag_modified


class SaveMetadata(NodeBase):

    def __init__(self, loggers=[]):
        self.loggers = loggers

    def __call__(self, input=None):
        for obj in input:
            yield obj

        # export logging information
        job = get_current_job()
        if job.get_id():
            logs = {}
            for log in self.loggers:
                logs.update(log.get_log())

            task = models.Task.query.filter(
                models.Task.id == job.get_id()).first()

            meta = json.loads(task.meta)
            meta.update(logs)
            task.meta = json.dumps(meta)

            database.session.commit()
