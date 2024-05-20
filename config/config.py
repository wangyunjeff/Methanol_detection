from torch.utils.tensorboard import SummaryWriter


# class GeneralConfig:
#     log_dir = 'runs_SIN_difTest/MLP-hidden20-LR0.001'
#     batch_size = 64
#     num_epochs = 100
#     train_data_path = './SIN_train_data_with_labels_difTest.csv'
#     test_data_path = './SIN_test_data_with_labels_difTest.csv'
#
# class ModelConfig:
#     hidden_size = 20
#     learning_rate = 0.001
#     dropout_rate = 0.0
#
# # Initialize SummaryWriter separately
# writer = SummaryWriter(GeneralConfig.log_dir)
#
# class Config:
#     general = GeneralConfig()
#     model = ModelConfig()
#     writer = writer
class Config:
    def __init__(self):
        self.general = self.GeneralConfig()
        self.model = self.ModelConfig()
        self.training = self.TrainingConfig()

    class GeneralConfig:

        def __init__(self):
            self.log_dir = 'runs_SIN_difTest/MLP-hidden20-LR0.001'
            self.batch_size = 64
            self.num_epochs = 100
            self.train_data_path = './SIN_train_data_with_labels_difTest.csv'
            self.test_data_path = './SIN_test_data_with_labels_difTest.csv'
            self.data_loader_name = "default"  # default triplet triplet_with_exposed
            self.loss_name = "CrossEntropyLoss"  # CrossEntropyLoss
            self.optimizer_name = "Adam"
            # self.train_data_path = "path/to/train/data.csv"
            # self.test_data_path = "path/to/test/data.csv"
            # self.batch_size = 32
            # self.num_epochs = 50
            # self.model_name = "SimpleMLP"  # Name of the model to use
            # self.loss_name = "CrossEntropyLoss"  # Name of the loss function

    class ModelConfig:
        def __init__(self):
            self.model_name = 'mlp'  # mlp SimpleMLP_triplet
            self.hidden_size = 20
            self.dropout_rate = 0.5
            self.learning_rate = 0.001

    class TrainingConfig:
        def __init__(self):
            self.optimizer_name = "Adam"  # Name of the optimizer to use
            self.train_method = 'default' # default triplet

config = Config()
if __name__ == '__main__':
    # Accessing hidden_size
    hidden_size = config.model.hidden_size

    print(hidden_size)  # Output: 20
