class Policy:
    def act(self, observation):
        raise NotImplementedError

    def learn(self, batch):
        raise NotImplementedError
