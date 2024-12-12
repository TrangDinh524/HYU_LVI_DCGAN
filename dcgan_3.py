# %%
import os
import os.path as osp
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import DataLoader
from tqdm import tqdm
import time


# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results


# %%
# Root directory for dataset
dataroot = "./data"
save_dir = "./dcgan_4"
os.makedirs(save_dir, exist_ok=True)
compute_intra_fid = True

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
eval_every = 1
num_epochs = 100

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# %%
# We can use an image folder dataset the way we have it setup.
# Create the dataset
transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
dataset = dset.CIFAR100(root='./data', train=True, download=True, transform=transform)
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Take 64 fixed samples for FID
fixed_batch_for_fid = next(iter(dataloader))
fixed_batch_for_fid = fixed_batch_for_fid[0][:64].to(device)

# Plot some training images
# real_batch = next(iter(dataloader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
# plt.show()

# %%
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.linalg import sqrtm

superclass_mapping = {
    0: [4, 30, 55, 72, 95],  # aquatic mammals
    1: [1, 32, 67, 73, 91],  # fish
    2: [54, 62, 70, 82, 92],  # flowers
    3: [9, 10, 16, 28, 61],  # food containers
    4: [0, 51, 53, 57, 83],  # fruit and vegetables
    5: [22, 39, 40, 86, 87],  # household electrical devices
    6: [5, 20, 25, 84, 94],  # household furniture
    7: [6, 7, 14, 18, 24],  # insects
    8: [3, 42, 43, 88, 97],  # large carnivores
    9: [12, 17, 37, 68, 76],  # large man-made outdoor things
    10: [23, 33, 49, 60, 71],  # large natural outdoor scenes
    11: [15, 19, 21, 31, 38],  # large omnivores and herbivores
    12: [34, 63, 64, 66, 75],  # medium-sized mammals
    13: [26, 45, 77, 79, 99],  # non-insect invertebrates
    14: [2, 11, 35, 46, 98],  # people
    15: [27, 29, 44, 78, 93],  # reptiles
    16: [36, 50, 65, 74, 80],  # small mammals
    17: [47, 52, 56, 59, 96],  # trees
    18: [8, 13, 48, 58, 90],  # vehicles 1
    19: [41, 69, 81, 85, 89]   # vehicles 2
}


def load_inception_net():
  inception_model = inception_v3(pretrained=True, transform_input=False)
  inception_model = WrapInception(inception_model.eval()).cuda()
  return inception_model


class WrapInception(nn.Module):
    def __init__(self, net):
        super(WrapInception,self).__init__()
        self.net = net
        self.mean = P(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
                    requires_grad=False)
        self.std = P(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
                    requires_grad=False)
    def forward(self, x):
        # Normalize x
        x = (x + 1.) / 2.0
        x = (x - self.mean) / self.std
        # Upsample if necessary
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
        # 299 x 299 x 3
        x = self.net.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.net.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.net.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.net.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.net.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.net.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.net.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.net.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.net.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6e(x)
        # 17 x 17 x 768
        # 17 x 17 x 768
        x = self.net.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.net.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.net.Mixed_7c(x)
        # 8 x 8 x 2048
        pool = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
        # 1 x 1 x 2048
        logits = self.net.fc(F.dropout(pool, training=False).view(pool.size(0), -1))
        # 1000 (num_classes)
        return pool, logits


def get_net_output(train_loader, net,device):
    net.eval()
    pool, logits, labels = [], [], []

    for i, (x, y) in enumerate(tqdm(train_loader, desc="Get net output")):
        x = x.to(device)
        with torch.no_grad():
            pool_val, logits_val = net(x)
            pool += [np.asarray(pool_val.cpu())]
            logits += [np.asarray(F.softmax(logits_val, 1).cpu())]
            labels += [np.asarray(y.cpu())]
    pool, logits, labels = [np.concatenate(item, 0) for item in [pool, logits, labels]]
    net.train()
    return pool, logits, labels


def accumulate_inception_activations(sample, net, num_inception_images=50000):
    pool, logits, labels = [], [], []
    i = 0
    while (torch.cat(logits, 0).shape[0] if len(logits) else 0) < num_inception_images:
        with torch.no_grad():
            images, labels_val = sample()
            pool_val, logits_val = net(images.float())
            pool += [pool_val]
            logits += [F.softmax(logits_val, 1)]
            labels += [labels_val]
    return torch.cat(pool, 0), torch.cat(logits, 0), torch.cat(labels, 0)


def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def calculate_inception_score(pred, num_splits=10):
    scores = []
    for index in range(num_splits):
        pred_chunk = pred[index * (pred.shape[0] // num_splits): (index + 1) * (pred.shape[0] // num_splits), :]
        kl_inception = pred_chunk * (np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
        kl_inception = np.mean(np.sum(kl_inception, 1))
        scores.append(np.exp(kl_inception))
    return np.mean(scores), np.std(scores)



def calculate_intra_fid(pool, logits,labels ,g_pool, g_logits, g_labels, chage_superclass=True):
    intra_fids = []
    super_class = super_class_mapping()

    super_labels = [super_class[i] for i in labels]
    super_labels = np.array(super_labels)

    if chage_superclass:
        g_super_labels = [super_class[i] for i in g_labels]
        g_super_labels = np.array(g_super_labels)
    else:
        g_super_labels = np.array(g_labels.cpu())

    for super_idx, _ in superclass_mapping.items():
        mask = (super_labels == super_idx)
        g_mask = (g_super_labels == super_idx)

        pool_low = pool[mask]
        g_pool_low = g_pool[g_mask]

        assert 2500 == len(g_pool_low), "super-classes count error"
        if len(pool_low) == 0 or len(g_pool_low) == 0:
            continue

        mu, sigma = np.mean(g_pool_low, axis=0), np.cov(g_pool_low, rowvar=False)
        mu_data, sigma_data = np.mean(pool_low, axis=0), np.cov(pool_low, rowvar=False)

        fid = calculate_fid(mu, sigma, mu_data, sigma_data)
        intra_fids.append(fid)

    return np.mean(intra_fids), intra_fids


def super_class_mapping():
    class_to_superclass = [None] * 100
    for super_idx, class_indices in superclass_mapping.items():
        for class_idx in class_indices:
            class_to_superclass[class_idx] = super_idx
    return class_to_superclass

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# %%
# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)

# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
#  to ``mean=0``, ``stdev=0.02``.
netG.apply(weights_init)

# Print the model
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
# like this: ``to mean=0, stdev=0.2``.
netD.apply(weights_init)

# Print the model
print(netD)



# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# %%
class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, size, generator, num_classes):
        self.size = size
        assert size % num_classes == 0, "Size must be divisible by num_classes"
        self.num_classes = num_classes
        self.generator = generator

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        with torch.no_grad():
            return self.generator(torch.randn(1, nz, 1, 1, device=device)).squeeze(0), idx % self.num_classes

# %%
# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
fid_scores = []
intra_fid_scores = []
is_list = []
iters = 0
total_train_time = 0.0
total_eval_time = 0.0
inception_net = load_inception_net()

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader

    real_images_concat = []
    fake_images_concat = []
    _start_time = time.time()
    for i, data in enumerate(tqdm(dataloader, desc="Training loop")):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        real_images_concat.append(real_cpu)
        fake_images_concat.append(fake)

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        iters += 1

    train_time = time.time() - _start_time
    total_train_time += train_time
    _start_time = time.time()

    if epoch % eval_every == 0 and epoch > 0:
        # print("concat", len(real_images_concat), len(fake_images_concat))
        with torch.no_grad():
            real_features, _, real_labels = get_net_output(dataloader,inception_net,device)
            fake_dataset = FakeDataset(len(dataset), netG, 100)
            fake_dataloader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=True)
            fake_features, fake_preds, fake_labels = get_net_output(fake_dataloader,inception_net,device)
        # print(f"Real dataset size: {len(real_dataset)}")
        # print(f"Real dataloader batches: {len(dataloader)}")
        # print(f"Fake dataset size: {len(fake_dataset)}")
        # print(f"Fake dataloader batches: {len(fake_dataloader)}")
        img_list.append(vutils.make_grid(fake.cpu(), padding=2, normalize=True))


        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
        # Calculate FID
        fid_score = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
        fid_scores.append((iters,fid_score))
        # Calculate intra FID
        if compute_intra_fid:
            intra_fid, _ = calculate_intra_fid(real_features, None, real_labels, fake_features, None, fake_labels)
            intra_fid_scores.append(intra_fid)
        # Calculate Inception Score
        is_value, is_std = calculate_inception_score(fake_preds)
        is_list.append(is_value)

        eval_time = time.time() - _start_time
        total_eval_time += eval_time
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tFID: %.2f\tIntra FID: %.2f\tIS: %.2f\tTrain Time: %.2f\tEval Time: %.2f'
            % (
                epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2,
                fid_score, intra_fid if compute_intra_fid else 0.0, is_value, train_time, eval_time
            )
        )

print(f"Total training time: {total_train_time:.2f}s")
print(f"Total evaluation time: {total_eval_time:.2f}s")

# %%
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(osp.join(save_dir, "loss_plot.png"))
plt.close()

# plt.figure(figsize=(10,5))
# plt.title("FID Scores During Training")
# fid_steps_, fid_scores_ = zip(*fid_scores)
# plt.plot(fid_steps_, fid_scores_)
# plt.xlabel("iterations")
# plt.ylabel("FID")
# plt.legend()
# plt.show()

# Plot metrics
plt.figure(figsize=(10, 5))
fid_steps_, fid_scores_ = zip(*fid_scores)
plt.plot(fid_steps_, fid_scores, label="FID")
if compute_intra_fid:
    plt.plot(fid_steps_, intra_fid_scores, label="Intra FID")
plt.plot(fid_steps_, is_list, label="Inception Score")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.legend()
plt.savefig(osp.join(save_dir, "metrics_plot.png"))
plt.close()


#Real Images vs. Fake Images

# # Grab a batch of real images from the dataloader
# real_batch = next(iter(dataloader))

# # Plot the real images
# plt.figure(figsize=(15,15))
# plt.subplot(1,2,1)
# plt.axis("off")
# plt.title("Real Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# # Plot the fake images from the last epoch
# plt.subplot(1,2,2)
# plt.axis("off")
# plt.title("Fake Images")
# plt.imshow(np.transpose(img_list[-1],(1,2,0)))
# plt.show()



#Visualization of G's progression

def animate_numpy_arrays(arrays, fps=1):
    fig, ax = plt.subplots(figsize=(8,8))
    image = ax.imshow(arrays[0])
    ax.axis('off')  # Hide axes

    def animate(i):
        image.set_data(arrays[i])
        ax.set_title(f"Epoch = {i}")
        return image,  # Return the updated image object

    ani = animation.FuncAnimation(fig, animate, frames=len(arrays), interval=1000/fps, blit=True)

    # Save as GIF using imageio
    writer = animation.PillowWriter(fps=fps,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
    ani.save(osp.join(save_dir, 'progress.gif'), writer=writer)

arrays = [np.transpose(i, (1,2,0)) for i in img_list]
animate_numpy_arrays(arrays, fps=1)

