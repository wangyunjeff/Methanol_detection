import torch.nn as nn

def get_loss_function(loss_name):
    if loss_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    if loss_name == "TripletLoss":
        return nn.TripletMarginLoss(margin=1.0, p=2)
    # Add other loss functions here
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
