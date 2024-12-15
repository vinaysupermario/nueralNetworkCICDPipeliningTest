# Neural Network CI/CD Pipeline

## The pipeline will:

1. Train the model for 1 epoch
2. Run tests checking:
    - Model architecture (input shape 28x28, output shape 10)
    - Parameter count (< 25000)
    - Model accuracy (> 95% on test set)
3. Save the trained model with timestamp
4. Upload the model as an artifact in GitHub Actions

## The model architecture is a simple CNN with:
 - 2 convolutional layers
 - 2 max pooling layers
 - 2 fully connected layers
 - ReLU activation functions

**The saved model file will have a timestamp suffix (e.g., model_20240314_153022.pth) for tracking when it was trained.**