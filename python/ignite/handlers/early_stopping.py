import logging

from ignite.engine import Engine


class EarlyStopping(object):
    """EarlyStopping handler can be used to stop the training if no improvement after a given number of events

    Args:
        patience (int):
            Number of events to wait if no improvement and then stop the training
        score_function (Callable):
            It should be a function taking a single argument, an `ignite.engine.Engine` object,
            and return a score `float`. An improvement is considered if the score is higher.

    Examples:

    .. code-block:: python

        from ignite.engine import Engine, Events
        from ignite.handlers import EarlyStopping

        def score_function(engine):
            val_loss = engine.state.metrics['nll']
            return -val_loss

        handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, handler)

    """
    def __init__(self, patience, score_function, trainer):
        assert callable(score_function), "Argument score_function should be a function"
        assert patience > 0, "Argument patience should be positive"
        assert isinstance(trainer, Engine), "Argument trainer should be an instance of Engine"
        self.score_function = score_function
        self.patience = patience
        self.trainer = trainer
        self.counter = 0
        self.best_score = None
        self._logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._logger.addHandler(logging.NullHandler())

    def __call__(self, engine):
        score = self.score_function(engine)

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            self._logger.debug("EarlyStopping: %i / %i" % (self.counter, self.patience))
            if self.counter >= self.patience:
                self._logger.info("EarlyStopping: Stop training")
                self.trainer.terminate()
        else:
            self.best_score = score
            self.counter = 0
