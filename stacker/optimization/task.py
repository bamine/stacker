from sklearn import model_selection


class Task:
    def __init__(self, name, X, y, task, test_size=None, cv=None, random_state=42):
        self.name = name
        self.X = X
        self.y = y
        self.task = task
        self.random_state = random_state
        if test_size is not None:
            self.test_size = test_size
            self.validation_method = "train_test_split"
            self.X_train, self.X_test, self.y_train, self.y_test = \
                model_selection.train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        elif cv is not None:
            self.validation_method = "cv"
            if task == "regression":
                self.kfold = model_selection.KFold(n_splits=cv, random_state=random_state)
            elif task == "classification":
                self.kfold = model_selection.StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    def __str__(self):
        return "Task={0} - type={1} - validation_method={2} - len(X)={3}"\
            .format(self.name, self.task, self.validation_method, len(self.X))