import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomMidiLoss(nn.Module):
    def __init__(self, weight_classification=1.0, weight_regression=1.0):
        super(CustomMidiLoss, self).__init__()
        self.weight_classification = weight_classification
        self.weight_regression = weight_regression

    def forward(self, predicted_output, target):
        #Amount of padding is to match target with predicted so that we do not need to truncate

        padding_size = len(predicted_output) - len(target)
        print(f'target: {target}')
        #Apply padding to the target tensor
        padding_size = 2

        # Get the shape of the target tensor
        target_shape = target.size()

        # Create zeros tensor with appropriate shape
        zeros_tensor = torch.zeros((padding_size,) + target_shape)

        # Concatenate target with zeros tensor along the first dimension
        concatenated_target = torch.cat([target.unsqueeze(0), zeros_tensor], dim=0)
        print(f'concatentated target: {concatenated_target}')
        
        #split the padded target into discrete and continuous values
        #one hot encoding for discrete values
        one_hot_encoding = torch.where(concatenated_target[:, 0:1] >= 0.5, torch.tensor(1), torch.tensor(0))

        target_classification = torch.sum(one_hot_encoding.float(), dim = 0) #changing into float for mse --> pitch
        target_regression = concatenated_target[:, 1:2] #second column --> time
        target_regression2 = concatenated_target[:, 2:3] #third column --> duration

        #Split the predicted output in the same way
        predicted_classification = predicted_output[0:1] #first column --> pitch
        predicted_regression = predicted_output[1:2] #second column --> time
        predicted_regression2 = predicted_output[2:3] #third column --> duration

        print(f'predicted classification: {predicted_classification}')
        print(f'target classification: {target_classification}')
        
        #Cross Entropy Loss for Pitch
        loss_classification = F.cross_entropy(predicted_classification.flatten(), target_classification.flatten())

        #Mean Squared Error for Time
        loss_regression = F.smooth_l1_loss(predicted_regression.flatten(), target_regression.flatten())

        #Mean Squared Error for Duration
        loss_regression2 = F.smooth_l1_loss(predicted_regression2.flatten(), target_regression2.flatten())

        print(f'loss classification: {loss_classification}')
        print(f' loss regression: {loss_regression}')
        print(f' loss regression2:{loss_regression2}')

        #Combine the losses with weights
        total_loss = (
            self.weight_classification * loss_classification +
            self.weight_regression * loss_regression +
            self.weight_regression * loss_regression2
        )

        return total_loss