class SingletonConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SingletonConfig, cls).__new__(cls)
            cls._instance.debug = []
        return cls._instance
