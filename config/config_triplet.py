from torch.utils.tensorboard import SummaryWriter


class Config:
    def __init__(self):
        self.general = self.GeneralConfig()
        self.model = self.ModelConfig()
        self.training = self.TrainingConfig()

    class GeneralConfig:

        def __init__(self):
            self.log_dir = 'runs_SIN_difTest/MLP-hidden20-LR0.001'
            self.batch_size = 64
            self.num_epochs = 10000
            self.train_data_path = './SIN_train_data_with_labels_difTest.csv'
            self.test_data_path = './SIN_test_data_with_labels_difTest.csv'
            self.data_loader_name = "Triplet"  # default triplet triplet_with_exposed
            self.loss_name = "TripletLoss"  # CrossEntropyLoss TripletLoss
            self.optimizer_name = "Adam"

    class ModelConfig:
        def __init__(self):
            self.model_name = 'SimpleMLP_triplet'  # mlp SimpleMLP_triplet
            self.hidden_size = 100
            self.dropout_rate = 0.0
            self.learning_rate = 0.001

    class TrainingConfig:
        def __init__(self):
            self.optimizer_name = "Adam"  # Name of the optimizer to use
            self.train_method = 'Triplet'  # default

config = Config()
