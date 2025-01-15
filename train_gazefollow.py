import os
import yaml
import torch
import numpy as np
from torch.optim import RMSprop, Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tqdm
from network.network_builder import build_lstm


def main():
    # Load config file
    with open('configuration.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # load config
    lr = config['optimizer']['lr']
    num_epochs = config['training']['epochs']
    early_stop_patience = config['training']['early_stop_patience']

    model, criteria = build_lstm(config)

    if config['optimizer']['name'] == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr=lr)
    elif config['optimizer']['name'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr)
    elif config['optimizer']['name'] == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=lr)
    else:
        raise TypeError("Optimizer not supported!")

    # Define a linear learning rate scheduler
    lr_step_size = config['optimizer']['lr_step_size']
    gamma = config['optimizer']['gamma']  # Factor by which the learning rate will be reduced
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

    train_loader = build_data_loader(config)
    val_loader = build_data_loader(config, 'val')
    #test_loader = build_data_loader(config, 'test')

    seq_length = int(config['model']['seq_length_s'] / config['model']['samp_interval_s']) + 1

    # Custom loss function considering weights of all timesteps
    if config['training']['custom_weighted_loss'] == "softmax":
        seq_weights = torch.arange(seq_length, dtype=torch.float32)
        m = torch.nn.Softmax(dim=0)
        seq_weights = m(seq_weights)
    elif config['training']['custom_weighted_loss'] == "linear":
        seq_weights = torch.arange(seq_length, dtype=torch.float32)
        seq_weights = seq_weights / torch.sum(seq_weights)
    else:
        raise TypeError("Custom weighted loss type incorrect!")
    #print(seq_weights)

    best_val_loss = float('inf')
    early_stop_count = 0

    for epoch in range(num_epochs):
        print("\nEpoch No. {}:".format(epoch))

        epoch_loss = 0.0
        total = len(train_loader)

        with tqdm.tqdm(total=total) as pbar:
            # Shape of inputs: (batch_size, sequence_length, num_features)
            # Shape of targets: (batch_size, sequence_length, 1)
            for idx, (inputs, targets) in enumerate(train_loader):
                model.train()
                criteria.train()

                pred = model(inputs)

                #print("Out shape: ", pred.shape)

                # discuss if Loss func considers all timesteps or not
                if config['training']['is_only_last_timestep'] == "true":
                    loss = criteria(pred[:, -1], targets[:, -1])
                else:
                    loss = 0
                    for i in range(seq_length):
                        loss += criteria(pred[:, i, :], targets[:, i, :]) * seq_weights[i]

                loss_value = loss.item()
                epoch_loss += loss_value

                # Backward pass and optimization (for training)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(1)

        print("Training loss: ", epoch_loss / total)

        ### Eval ###
        model.eval()
        criteria.eval()

        val_loss = 0.0
        val_total = len(val_loader)

        with tqdm.tqdm(total=val_total) as pbar:
            with torch.no_grad():
                for idx, (inputs, targets) in enumerate(val_loader):

                    val_pred = model(inputs)

                    # discuss if Loss func considers all timesteps or not
                    if config['training']['is_only_last_timestep'] == "true":
                        loss = criteria(val_pred[:, -1], targets[:, -1])
                    else:
                        loss = 0
                        for i in range(seq_length):
                            loss += criteria(val_pred[:, i, :], targets[:, i, :]) * seq_weights[i]

                    val_loss += loss.item()
                    pbar.update(1)

            print("Validation loss: ", val_loss / val_total)

        # Example early stopping based on validation loss
        if val_loss / val_total < best_val_loss:
            best_val_loss = val_loss / val_total
            early_stop_count = 0
            # Save the best model if needed
            if not os.path.exists(config['training']['saved_dir']):
                os.makedirs(config['training']['saved_dir'])

            saved_file_name = config['training']['saved_best_model'].join(
                ("best",
                 config['model']['name'] + "_" + str(config['model']['hidden_dim1']) + "_" + str(config['model']['hidden_dim2']),
                 config['data']['pkl_name'].split('.')[-2].split('-')[-1],
                 "window" + str(config['model']['seq_length_s']) + "sec",
                 config['optimizer']['name'],
                 "batch" + str(config['training']['batch_size']) + ".pth"
                 )
            )
            torch.save(model, os.path.join(config['training']['saved_dir'], saved_file_name))
        else:
            early_stop_count += 1
            if early_stop_count >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}. Best validation loss: {best_val_loss}")
                break

        scheduler.step()


if __name__ == "__main__":
    main()
