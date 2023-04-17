from globals import *
from utils.utils import *
from utils.visualisation import show_evaluation

from model import *

if __name__ == "__main__":
    #加载数据集
    dataset, scaler = load_dataset(config.dataset_path, show_data=True)
    # Prepare the dataset for training/testing
    subsequences = extract_subsequences(dataset, lag=config.lag)
    # Split the dataset into train/test set
    train_loader, test_set = train_test_split(subsequences)

    #训练模式
    if config.mode == "train":
        # Create new instance of the RNN
        net = LSTMModel(input_dim=train_loader.dataset[0][0].shape[-1],
                          hidden_dim=config.hidden_dim,
                          num_layers=config.num_layers)
        net.to(device)
        #训练模型
        train_loop(net, config.epochs, config.lr, config.wd,train_loader, test_set, debug=True)
    else:
        # Create new instance of the RNN using default values
        net = LSTMModel(input_dim=train_loader.dataset[0][0].shape[-1],
                          hidden_dim=parser.get_default('hidden_dim'),
                          num_layers=parser.get_default('num_layers'))
        net.to(device)
        # Load pretrained weights
        net.load_state_dict(torch.load(
            config.pretrained_path, map_location=device))

    # Display the prediction next to the target output values
    show_evaluation(net, dataset, scaler, debug=True)
