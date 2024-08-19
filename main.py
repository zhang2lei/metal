from train.train import train_model
from test.test import test_model

if __name__ == "__main__":
    data_path = "path/to/data"
    model_path = "specmet_model.pth"
    epochs = 100
    batch_size = 32
    lr = 1e-4

    # Train the model
    train_model(data_path, epochs, batch_size, lr)

    # Test the model
    test_model(data_path, model_path)
