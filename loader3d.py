import torch
from networks import RED_CNN  # Import your RED-CNN model definition
from loader import get_loader

# Set device and other training parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Load your data using the data loader
train_loader = get_loader(mode='train', saved_path='/projects/synergy_lab/garvit217/data_3d_2/train/4/HQ/', 
                        batch_size=batch_size, num_workers=6)

# Instantiate your RED-CNN model
red_cnn_model = RED_CNN().to(device)

# Set up optimizer and loss function
optimizer = torch.optim.Adam(red_cnn_model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (input_data, target_data) in enumerate(train_loader):
        # Move data to the device (CPU or GPU)
        input_data, target_data = input_data.to(device), target_data.to(device)

        # Forward pass
        output = red_cnn_model(input_data)

        # Compute loss
        loss = criterion(output, target_data)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training information if needed
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')

# Save the trained model if needed
torch.save(red_cnn_model.state_dict(), 'red_cnn_model.pth')
