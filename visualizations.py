from src.factory import *
import torchvision.transforms as ttf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import argparse
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import re

class PlotParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--root_dir', required=False, help='Root directory of the dataset')
        self.parser.add_argument('--batch_size', required=False, type=int, default=32, help='input batch size')
        self.parser.add_argument('--backbone', required=False, type=str, default='vgg16', help='which architecture to use. [resnet50, resnet152, resnext, vgg16]')
        self.parser.add_argument('--pool', required=False, type=str, default='GeM', help='Global pool layer  max|avg|GeM')
        self.parser.add_argument('--p', required=False, type=int, default=3, help='P parameter for GeM pool')
        self.parser.add_argument('--norm', required=False, type=str, default="L2", help='Norm layer')
        self.parser.add_argument('--image_size', required=False, type=str, default="480,640", help='Input size, separated by commas')
        self.parser.add_argument('--last_layer', required=False, type=int, default=None, help='Last layer to keep')
        self.parser.add_argument('--model_file', required=False, type=str, help='Model weights')
        self.parser.add_argument('--name', required=False, type=str, help='Name of the model')
        self.parser.add_argument('--log_file', required=False, type=str, help='Log file')

    def parse(self):
        self.opt = self.parser.parse_args()



name_mapping = {'CL': 'Contrastive',
                'GCL': 'Generalized Contrastive',
                'triplet_GCL': 'Triplet Generalized Contrastive',
                'contrastive_adaptive': 'Adaptive Margin Contrastive',
                'triplet': 'Triplet',
                'triplet_adaptive': 'Adaptive Margin Triplet',
                'SARE': 'SARE',
                'triplet_GCL3': 'Triplet Generalized Contrastive 2'
}


def plot_tsne(params):
    seed_value = 0
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)

    image_size = [int(x) for x in (params.image_size).split(",")]

    if image_size[0] == image_size[1]: #If we want to resize to square, we do resize+crop
        image_t = ttf.Compose([ttf.Resize(size=(image_size[0])),
                                ttf.CenterCrop(size=(image_size[0])),
                                ttf.ToTensor(),
                                ttf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
    else:
        image_t = ttf.Compose([ttf.Resize(size=(image_size[0], image_size[1])),
                                ttf.ToTensor(),
                                ttf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
        
    dataloader = create_msls_dataloader('soft_MSLS', params.root_dir, 'val',
                                        transform=image_t,
                                        batch_size=params.batch_size)

    model = create_model(params.backbone, params.pool, last_layer=params.last_layer, norm=params.norm, p_gem=params.p,
                         mode='siamese')
    
    try:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(params.model_file)["model_state_dict"])
        else:
            model.load_state_dict(
                torch.load(params.model_file, map_location=torch.device('cpu'))["model_state_dict"])
    except:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(params.model_file)["state_dict"])
        else:
            model.load_state_dict(torch.load(params.model_file, map_location=torch.device('cpu'))["state_dict"])

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    pos_pairs = list()
    neg_pairs = list()

    for i, data in enumerate(tqdm(dataloader)):
        if len(neg_pairs) >= 1000 and len(pos_pairs) >= 1000:
            break

        for j in range(data["label"].size(0)):  # Loop over samples in batch
            if (data["label"][j] >= 0.5 and len(pos_pairs) < 1000) or (data["label"][j] < 0.5 and len(neg_pairs) < 1000):
                if torch.cuda.is_available(): 
                    x0, x1 = model(data["im0"][j].unsqueeze(0).cuda(), data["im1"][j].unsqueeze(0).cuda())
                else:
                    x0, x1 = model(data["im0"][j].unsqueeze(0), data["im1"][j].unsqueeze(0))

                diff = (x0 - x1).cpu().detach().numpy()  # compute difference vector

                if (data["label"][j] >= 0.5):
                    pos_pairs.append(diff)
                else:
                    neg_pairs.append(diff)

    pos_pairs = np.array(pos_pairs)
    neg_pairs = np.array(neg_pairs)

    pos_labels = np.ones(pos_pairs.shape[0])
    neg_labels = np.zeros(neg_pairs.shape[0])

    all_pairs = np.concatenate((pos_pairs, neg_pairs))
    all_labels = np.concatenate((pos_labels, neg_labels))

    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(all_pairs.reshape(len(all_pairs), -1))

    plt.figure(figsize=(6, 5))
    colors = ['tab:blue', 'tab:orange']
    labels = ['Negative pairs', 'Positive pairs']

    for i, color, label in zip(range(2), colors, labels):
        plt.scatter(tsne_results[all_labels == i, 0], tsne_results[all_labels == i, 1], c=color, label=label, s=5)

    plt.legend()
    plt.savefig(f'/home/s2288176/acvpr/tsnes/{params.name.split("/")[-1].rsplit(".", 1)[0]}.png', bbox_inches='tight')
    plt.show()

# Function to plot saliency maps for a few example images
def example_saliency_maps(params, num_images=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = [int(x) for x in (params.image_size).split(",")]

    if image_size[0] == image_size[1]: #If we want to resize to square, we do resize+crop
        image_t = ttf.Compose([ttf.Resize(size=(image_size[0])),
                                ttf.CenterCrop(size=(image_size[0])),
                                ttf.ToTensor(),
                                ttf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
    else:
        image_t = ttf.Compose([ttf.Resize(size=(image_size[0], image_size[1])),
                                ttf.ToTensor(),
                                ttf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
    """
    weights = [os.path.join('models', 'MSLS', f'MSLS_{params.backbone}_GeM_480_CL.pth'),
               os.path.join('models', 'MSLS', f'MSLS_{params.backbone}_GeM_480_GCL.pth'),
               os.path.join('models', 'MSLS', f'MSLS_{params.backbone}_GeM_480_triplet_GCL.pth'),
               os.path.join('models', 'MSLS', f'MSLS_{params.backbone}_GeM_480_contrastive_adaptive.pth'),
               os.path.join('models', 'MSLS', f'MSLS_{params.backbone}_GeM_480_triplet.pth'),
               os.path.join('models', 'MSLS', f'MSLS_{params.backbone}_GeM_480_triplet_adaptive.pth'),
               os.path.join('models', 'MSLS', f'MSLS_{params.backbone}_GeM_480_SARE.pth'),
               ]
    """
    weights = params.model_file

    qfile = params.root_dir +"train_val/"+ "cph" + "/query.json"

    ds = TestDataSet(params.root_dir, qfile, transform=image_t)
    dataloader = DataLoader(ds, batch_size=num_images, shuffle=True)

    fig, axs = plt.subplots(nrows=(num_images), ncols=len(weights) + 1, figsize=(4*(len(weights) + 1), 
                                                                                    4*num_images))
    fig.subplots_adjust(hspace=0.025, wspace=0.05)
    
    for images in dataloader:
        for model_idx, weight in enumerate(weights):
            model = create_model(params.backbone, params.pool, last_layer=params.last_layer, norm=params.norm, p_gem=params.p,
                                mode='single')
        
            try:
                if torch.cuda.is_available():
                    model.load_state_dict(torch.load(weight)["model_state_dict"])
                else:
                    model.load_state_dict(
                        torch.load(weight, map_location=torch.device('cpu'))["model_state_dict"])
            except:
                if torch.cuda.is_available():
                    model.load_state_dict(torch.load(weight)["state_dict"])
                else:
                    model.load_state_dict(torch.load(weight, map_location=torch.device('cpu'))["state_dict"])

            model = model.to(device)

            for img_idx, image in enumerate(images):
                image = image.unsqueeze(0).to(device)

                image.requires_grad_() 

                outputs = model(image)

                #output_max_index = outputs.argmax()
                output_vals = outputs[0]
                mean_output_value = output_vals.mean()

                model.zero_grad()
                mean_output_value.backward()
                gradients = image.grad.data
                gradients = gradients.abs().max(dim=1)[0]

                # Obtain the min and max values at the 1st and 99th percentiles
                min_val, max_val = np.percentile(gradients.cpu().numpy(), 1), np.percentile(gradients.cpu().numpy(), 99)

                # Normalization to range [0, 1]
                gradients = (gradients - min_val) / (max_val - min_val)
                gradients = gradients.clamp(0, 1)

                gradients = gaussian_filter(gradients.cpu().numpy(), sigma=20)

                # Unnormalize the image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                image = std * image + mean
                image = np.clip(image, 0, 1)

                if weight == weights[0]:
                    axs[img_idx][0].imshow(image)
                    if img_idx == 0:
                        axs[img_idx][0].set_title('Original Image', fontsize=15)
                    axs[img_idx][0].axis('off')

                # Plot the image and the saliency map overlay
                axs[img_idx][model_idx + 1].imshow(image)
                axs[img_idx][model_idx + 1].imshow(gradients[0], cmap='jet', alpha=0.5)
                if img_idx == 0:
                    model_name = params.name.split("/")[-1].rsplit(".", 1)[0]
                    axs[img_idx][model_idx + 1].set_title(f'{model_name}', fontsize=15)

                axs[img_idx][model_idx + 1].axis('off')

        break
    plt.show()

def plot_loss_curves(params):
    with open(params.log_file, 'rb') as file:  # Open in binary mode
        binary_data = file.read()
        log_data = binary_data.decode('utf-8')  # Decode binary data to string

    # Step 2: Extract loss values using regular expressions
    pattern = r'Loss (\d+\.\d+)'  # Pattern to match 'Loss' followed by a float number
    loss_values = [float(match) for match in re.findall(pattern, log_data)]


    # Step 3: Plot the data
    plt.figure(figsize=(6, 5))
    plt.plot(loss_values)
    plt.title('Loss Value Over Iterations')
    plt.xlabel('Iteration Number')
    plt.ylabel('Loss Value')
    plt.savefig(f'/home/s2288176/acvpr/loss_curves/loss/{params.name.split("/")[-1].rsplit(".", 1)[0]}.png', bbox_inches='tight')
    plt.show()

def plot_null_loss_curves(params):
    with open(params.log_file, 'rb') as file:  # Open in binary mode
        binary_data = file.read()
        log_data = binary_data.decode('utf-8')  # Decode binary data to string

    # Step 2: Extract loss values using regular expressions
    pattern = r'Null loss (\d+\.\d+)'  # Pattern to match 'Loss' followed by a float number
    loss_values = [float(match) for match in re.findall(pattern, log_data)]


    # Step 3: Plot the data
    plt.figure(figsize=(6, 5))
    plt.plot(loss_values)
    plt.title('Loss Value Over Iterations')
    plt.xlabel('Iteration Number')
    plt.ylabel('Loss Value')
    plt.savefig(f'/home/s2288176/acvpr/loss_curves/null_loss/{params.name.split("/")[-1].rsplit(".", 1)[0]}.png', bbox_inches='tight')
    plt.show()

def loss_average_plots(params):
    from collections import defaultdict
    from matplotlib.ticker import MaxNLocator
    
    with open(params.log_file, 'rb') as file:  # Open in binary mode
        binary_data = file.read()
        data = binary_data.decode('utf-8')  # Decode binary data to string
    """
    # Extract step, iteration, and loss values from the input
    pattern = re.compile(r"Step (\d+), Iteration (\d+), Loss (\d+\.\d+), Null loss \d+\.\d+")
    step_data = defaultdict(list)

    for match in pattern.findall(data):
        step, iteration, loss = int(match[0]), int(match[1]), float(match[2])
        step_data[step].append((iteration, loss))

    # Calculate average loss per step and organize iterations for each step
    step_avg_losses = {}
    step_iteration_intervals = {}

    for step, values in step_data.items():
        iterations, losses = zip(*values)
        avg_loss = sum(losses) / len(losses)
        step_avg_losses[step] = avg_loss
        step_iteration_intervals[step] = iterations

    # Plotting: Average loss per step and individual losses within each step
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot average loss per step
    steps = sorted(step_avg_losses.keys())
    avg_losses = [step_avg_losses[step] for step in steps]
    ax.plot(steps, avg_losses, label='Average Loss per Step', marker='o')

    # Plot detailed losses for each step's iteration intervals
    for step in steps:
        iterations = step_iteration_intervals[step]
        losses = [loss for _, loss in step_data[step]]
        ax.plot([step] * len(iterations), losses, 'x', label=f'Losses for Step {step}' if step == steps[0] else '')

    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig(f'./loss_curves/{params.name.split("/")[-1].rsplit(".", 1)[0]}.png', bbox_inches='tight')
    plt.show()
    """
    # Extract step, iteration, and loss values from the input
    pattern = re.compile(r"Step (\d+), Iteration (\d+), Loss (\d+\.\d+), Null loss \d+\.\d+")
    step_data = defaultdict(list)

    # Organize data by step
    for match in pattern.findall(data):
        step, iteration, loss = int(match[0]), int(match[1]), float(match[2])
        step_data[step].append(loss)

    # Calculate the average loss per step
    step_avg_losses = {step: sum(losses) / len(losses) for step, losses in step_data.items()}

    # Prepare steps and average loss lists
    steps_sorted = sorted(step_avg_losses.keys())
    avg_losses_sorted = [step_avg_losses[step] for step in steps_sorted]

    # Plot only the average losses
    plt.figure(figsize=(10, 5))
    plt.plot(steps_sorted, avg_losses_sorted, marker='o', linestyle='-', color='blue', label='Average Loss per Step')
    plt.xlabel('Step')
    plt.ylabel('Average Loss')
    plt.title('Average Loss per Step')
    plt.legend()
    plt.xticks(steps_sorted)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(f'/home/s2288176/acvpr/loss_curves/average_loss/{params.name.split("/")[-1].rsplit(".", 1)[0]}.png', bbox_inches='tight')
    plt.show()

    



if __name__ == "__main__":
    p = PlotParser()
    p.parse()
    params = p.opt

    # python visualizations.py --root_dir ../MSLS/ --backbone resnet50 --pool GeM --last_layer 2 --model_file ./models/MSLS/MSLS_resnet50_GeM_480_GCL.pth
    # plot_tsne(params)
    # example_saliency_maps(params) 
    plot_loss_curves(params)
    loss_average_plots(params)
    plot_null_loss_curves(params)