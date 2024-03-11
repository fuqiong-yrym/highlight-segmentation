import torch
def save_model_checkpoint(model, cp_name):
    torch.save(model.state_dict(), os.path.join(working_dir, cp_name))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Load model from saved checkpoint
def load_model_from_checkpoint(model, ckp_path):
    return model.load_state_dict(
        torch.load(
            ckp_path,
            map_location=get_device(),
        )
    )

# Send the Tensor or Model (input argument x) to the right device
# for this notebook. i.e. if GPU is enabled, then send to GPU/CUDA
# otherwise send to CPU.
def to_device(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x.cpu()
    
def get_model_parameters(m):
    total_params = sum(
        param.numel() for param in m.parameters()
    )
    return total_params

def print_model_parameters(m):
    num_model_parameters = get_model_parameters(m)
    print(f"The Model has {num_model_parameters/1e6:.2f}M parameters")
# end if

def close_figures():
    while len(plt.get_fignums()) > 0:
        plt.close()
    # end while
# end def

def print_title(title):
    title_len = len(title)
    dashes = ''.join(["-"] * title_len)
    print(f"\n{title}\n{dashes}")
# end def

# Validation: Check if CUDA is available

def prepareList():
  base_dir = '/content/Train_subset/'
  with open('train.lst', "r") as f:
      lines = f.read().splitlines()
  n = len(lines) # 3017 for training
  splits = [lines[i].split() for i in range(n)]
  inputs = [splits[i][0] for i in range(n)]
  targets = [splits[i][1] for i in range(n)]
  img_paths = [base_dir + inputs[i] for i in range(n)]
  mask_paths = [base_dir + targets[i] for i in range(n)]
  return img_paths[:100], mask_paths[:100] # choose first 100 samples for try out
print(f"CUDA: {torch.cuda.is_available()}")
