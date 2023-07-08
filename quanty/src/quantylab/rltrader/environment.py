class Environment:
    RESOURCE_IDX = 3  # resource 위치

    def __init__(self, resource_data=None):
        self.resource_data = resource_data
        self.observation = None
        self.idx = -1

    def reset(self):
        self.observation = None
        self.idx = -1

    def observe(self):
        if len(self.resource_data) > self.idx + 1:
            self.idx += 1
            self.observation = self.resource_data[self.idx]
            return self.observation
        return None

    def get_resource(self):
        if self.observation is not None:
            return self.observation
        return None
