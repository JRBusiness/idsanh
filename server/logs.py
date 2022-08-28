import datetime
import uuid

class LogHandler:
    def __init__(self, app_name):
        self.app_name = app_name
    
    def event_dict(self, logger, log_method, event_dict):
        event_dict['dd.trace_id'] = 0
        event_dict['dd.span_id'] = 0
        event_dict['appenv'] = log_method
        event_dict['datetime'] = datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S")
        event_dict['unique_id'] = str(uuid.uuid4())
        event_dict['app_name'] = self.app_name
        return event_dict
    
