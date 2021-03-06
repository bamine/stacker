class OptimizationResult:
    def __init__(self, task, model, parameters, score, scorer_name, validation_method, predictions, random_state):
        self.task = task
        self.model = model
        self.parameters = parameters
        self.score = score
        self.scorer_name = scorer_name
        self.validation_method = validation_method
        self.predictions = predictions
        self.random_state = random_state

