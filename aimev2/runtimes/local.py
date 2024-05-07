class LocalRuntime:
    def __init__(self, config) -> None:
        self.config = config

    @property
    def disable_tqdm(self):
        return False

    def upload(self, e, logdir):
        pass

    def finish(self, logdir):
        pass
