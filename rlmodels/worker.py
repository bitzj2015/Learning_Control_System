import ray

@ray.remote
class worker(object):
    def __init__(self, env):
        self.env = env
