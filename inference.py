"""
The following is a simple example algorithm.

It is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

from pathlib import Path
import json
from glob import glob
import SimpleITK
import numpy as np

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        ("stacked-barretts-esophagus-endoscopy-images",): interface_0_handler,
    }[interface_key]

    # Call the handler
    return handler()


def interface_0_handler():
    # Read the input
    input_stacked_barretts_esophagus_endoscopy_images = load_image_file_as_array(
        location=INPUT_PATH / "images/stacked-barretts-esophagus-endoscopy",
    )
    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    # Optional: part of the Docker-container image: resources/
    # resource_dir = Path("/opt/app/resources")
    # with open(resource_dir / "some_resource.txt", "r") as f:
    #     print(f.read())

    """ Run your model here """
    # for demonstration we will use the timm classification model from the model directory
    from omegaconf import OmegaConf

    import torch
    from peft import LoraConfig, get_peft_model
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    from src.models import get_model, ClassificationModule
    
    exp_name = 'swin_big_lora'
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = 'cuda' if torch.cuda.is_available() else device
    
    print(exp_name, device)
    cfg = OmegaConf.load(RESOURCE_PATH / f'{exp_name}.yaml')
    # process the input
    test_transform = A.Compose([
        # preprocessing
        A.Resize(height=cfg.dataset.preprocessing.resize.size[0], 
                 width=cfg.dataset.preprocessing.resize.size[1]),
        A.Normalize(**cfg.dataset.preprocessing.normalize),
        ToTensorV2()
    ])
    

    # load the model
    my_backbone = get_model(cfg.model.model_name, num_classes=cfg.model.model_args.num_classes, weights=None)
    my_backbone._feature_hook_handle.remove()
    if cfg.model.get('use_lora', False):
        # Determine target modules based on model architecture
        # Only target LoRA-supported layers: Linear, Conv1d/2d/3d, Embedding, MultiheadAttention
        model_name = cfg.model.model_name.lower()
        if 'convnext' in model_name:
            modules_to_save = ['classifier']
            target_module_names = []
            for name, module in backbone.named_modules():
                # We target Linear layers that are inside the 'features.7' block
                if 'features.7' in name and isinstance(module, torch.nn.Linear):
                    target_module_names.append(name)
        elif 'swin' in model_name:
            modules_to_save = ['head']
            target_module_names = []
            for name, module in my_backbone.named_modules():
                # We target Linear layers that are inside the 'features.7' block
                if '.mlp.' in name and isinstance(module, torch.nn.Linear):
                    target_module_names.append(name)
                elif name.endswith(('qkv', 'proj')) and isinstance(module, torch.nn.Linear):
                    target_module_names.append(name)
        else:
            assert False, f"LoRA not yet supported for {model_name}"
        
        lora_config = LoraConfig(
            # task_type=TaskType.FEATURE_EXTRACTION,  # For image classification backbone
            inference_mode=True,
            r=cfg.model.lora.get('r', 16),  # LoRA rank
            lora_alpha=cfg.model.lora.get('alpha', 32),  # LoRA alpha
            lora_dropout=cfg.model.lora.get('dropout', 0.),  # LoRA dropout
            target_modules=target_module_names,  # Model-specific target modules
            modules_to_save=modules_to_save
        )
        my_backbone = get_peft_model(my_backbone, lora_config)
        print(f"LoRA applied to {model_name} with target modules: {lora_config.target_modules}")
        my_backbone.print_trainable_parameters()
    
    my_model = ClassificationModule(my_backbone, 
                                    optimizer_name=cfg.training.optimizer_name, optimizer_args=cfg.training.optimizer_args,
                                    loss_name=cfg.model.loss_name, loss_args=cfg.model.loss_args)
    my_model.load_state_dict(torch.load(RESOURCE_PATH / f'{exp_name}.ckpt', map_location=device)['state_dict'])
    _ = my_model.to(device).eval()

    # predict
    if len(input_stacked_barretts_esophagus_endoscopy_images.shape) == 3:
        input_stacked_barretts_esophagus_endoscopy_images = input_stacked_barretts_esophagus_endoscopy_images.unsqueeze(0)
    num_imgs = len(input_stacked_barretts_esophagus_endoscopy_images)
    batch_size = 8
    num_batchs = num_imgs // batch_size
    num_batchs = num_batchs + 1 if num_imgs % batch_size != 0 else num_batchs

    output_stacked_neoplastic_lesion_likelihoods = []

    for i in range(num_imgs):
        # batch_imgs = input_stacked_barretts_esophagus_endoscopy_images[i*batch_size: (i + 1)*batch_size]
        img =  input_stacked_barretts_esophagus_endoscopy_images[i]
        batch_imgs = test_transform(image=img)['image'].to(device).unsqueeze(0)
        batch_logits = my_model(batch_imgs)
        batch_probs = torch.nn.functional.softmax(batch_logits, dim=1).detach().cpu()[:, 1].tolist()
        output_stacked_neoplastic_lesion_likelihoods.extend(batch_probs)
        
    # Save your output
    write_json_file(
        location=OUTPUT_PATH / "stacked-neoplastic-lesion-likelihoods.json",
        content=output_stacked_neoplastic_lesion_likelihoods,
    )
    print('save done')

    return 0


def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))

0
def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
