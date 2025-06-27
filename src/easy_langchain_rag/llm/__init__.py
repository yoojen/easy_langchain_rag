class LLMConfig:
    def __init__(self, model: str, temperature: float, disable_streaming: bool, callbacks: list, **kwargs):
        """
        Initialize a LLMConfig object.

        Args:
            model: The model to use.
            api_key: The API key to use.
            temperature: The temperature to use for the model.
            disable_streaming: Whether to disable streaming.
            callbacks: A list of callbacks to use.
            kwargs: Additional keyword arguments.
        """
        self.model = model
        self.temperature = temperature
        self.disable_streaming = disable_streaming
        self.callbacks = callbacks
        self.kwargs = kwargs

    def to_dict(self):
        """
        Convert the LLMConfig object to a dictionary.

        Returns:
            A dictionary of the LLMConfig object's attributes.
        """
        initial_config = {
            "model": self.model,
            "temperature": self.temperature,
            "disable_streaming": self.disable_streaming,
            "callbacks": self.callbacks
        }

        # Update Dictionary configurations with other user specified settings
        if self.kwargs:
            initial_config.update(self.kwargs)

        return initial_config
