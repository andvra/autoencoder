import torch
import torch.optim as optim 
import torch.utils.data as utils
import numpy as np 
import matplotlib.pyplot as plt
from convnet import ConvNet
import dataloader
import torch.nn as nn 
import json
import time

def to_img(x, channels, n):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), channels, n, n)
    return x

def train(dataloader: utils.DataLoader, device: str):
    """ Perform training on the samples in the dataloader, one batch at a time.
    Training is performed by running each sample through the network and comparing
    the output with the input. Weights are updated to increase performance.
    """
    lossfn = nn.MSELoss()
    net.train()
    # Load all the batches (n images at once) and run through the network
    for data in dataloader:
      img, _ = data
      img = img.to(device)
      output = net(img)
      loss = lossfn(img, output)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

# Can't be more than 100% loss, so initially setting this
#   to 1 is safe.
min_loss = 1
def test(dataloader: utils.DataLoader, device:str):
    """ Plot a sample to demonstrated the current quality of the 
    autoencoder
    """
    global min_loss
    lossfn = nn.MSELoss()
    img_input, _ = next(iter(dataloader))
    img_input = img_input.to(device)
    img_output = net(img_input)
    loss = lossfn(img_input, img_output)
    if loss<min_loss:
      min_loss = loss
    print(f'Loss: {loss:.4f}: {min_loss:.4f}')
    # img1 = to_img(img_input.cpu().data, net.num_channels, net.img_size)
    # img2 = to_img(img_output.cpu().data, net.num_channels, net.img_size)
    # idx = np.random.randint(len(img_input))
    # plt.subplot(2,1,1)
    # plt.imshow(img1[idx][0])
    # plt.subplot(2,1,2)
    # plt.imshow(img2[idx][0])
    # plt.show()

if __name__=="__main__":
    dataset_map = {
        'mnist': dataloader.Datasets.MNIST,
        'cifar10': dataloader.Datasets.CIFAR10}
    with open('conf.json') as json_file:
        conf = json.load(json_file)
        print(conf)
        # Code size is interesting. That's the number of neurons used to connect
        #   the encoder with the decoder. Changing this value will affect
        #   quality of the output image. 
        code_size = conf.get('code_size', 10)
        dataset_name = conf.get('dataset', 'mnist').lower()
        dataset = dataset_map.get(dataset_name, dataloader.Datasets.MNIST)
        # Updating epochs will affect bth performance and quality. Try with some
        #   different values!
        epochs = conf.get('epochs', 5)
        batch_size = conf.get('batch_size', 128)
    # Check if there's a GPU available. Highly recommendend since it's orders
    #   of magnitude faster
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Try the code with some different datasets. Currently, MNIST and CIFAR10
    #   are supported, but you could easily add more in dataloader.py
    trainloader, testloader = dataloader.get_dataloader(dataset, batch_size)
    # We only handle square images for now. The image shape can be read from
    #   the shape of the elements in the dataloader. The shape of each batch
    #   in the dataloader is on the format:
    #   (num_images, num_channels, width, height)
    #   num_images is capped at batch_size. num_channels represent the number of
    #   colors channels in the image: usually 1 (gray-scale) or 3 (colored).
    #   Since we expect the width and height to be the same, we read the third
    #   shape value (width) as our image size
    first_batch = next(iter(trainloader))[0]
    img_size = first_batch.shape[2]
    num_colors = first_batch.shape[1]
    net = ConvNet(code_size, img_size, num_colors, device).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    t_start = time.time()
    t_last = time.time()
    for epoch in range(epochs):
        train(trainloader, device)
        test(testloader, device)
        print(f'{time.time()-t_start:.1f}s\t{time.time()-t_last:.1f}s\tDone running epoch {epoch+1}')
        t_last = time.time()
    # Save the learned weights for later use
    torch.save(net, f'static/{dataset_name}_state.pth')
    # We want to sample some values sent to the decoder. The reason is that
    #   we want to use this to define a range for each of the n nodes in the
    #   code that we can use in the front end. We do this by sending a batch
    #   through the encoder.
    code_input = net.encoder(first_batch).detach().numpy()
    np.save(f'static/{dataset_name}_code', code_input)