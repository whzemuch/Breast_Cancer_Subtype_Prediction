class TrainerRegistry:
    """Central registry for all trainers"""

    _registry = {}

    @classmethod
    def register(cls, name):
        """Decorator for registering trainer classes"""

        def decorator(trainer_class):
            if name.lower() in cls._registry:
                raise ValueError(f"Trainer {name} already registered")
            cls._registry[name.lower()] = trainer_class
            return trainer_class

        return decorator

    @classmethod
    def get_trainer(cls, name, **kwargs):  # Accept arbitrary keyword arguments
        name = name.lower()
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown trainer: {name}. Available: {available}")
        return cls._registry[name](**kwargs)  # Pass all arguments to constructor