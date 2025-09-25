from src.datasets import get_loader
from .train import TrainExperiment


class TestExperiment(TrainExperiment):

    experiment_name = "test"
    wandb_offline = True

    def __init__(self, **kwargs) -> None:

        super().__init__(continue_training=True, **kwargs)

    #==========Run==========
    def run(self):
        self.save = False
        losses_test = self.test_epoch()
        metrics_eval = self.evaluate(self.ini_epoch)
        log_dict = {**losses_test, **metrics_eval}
        self.wandb_run.log(log_dict)
        print("Epoch {}".format(self.ini_epoch))
        print('\n'.join(["{}:\t{}".format(k, v) for k, v in log_dict.items()])+'\n')