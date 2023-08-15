# minimal-ddim
Implementing a minimal version of DDIM on MNIST.

## How to use

### Training DDPM
This puts visualizations in visuals/ and checkpoints in checkpoints
`python train_ddpm.py`

### Sampling with DDIM
`python ddim.py --checkpoint <path to checkpoint>`


### Results

The cool thing about DDIM is you can verify whether it works by checking whether applying forward and then backwards diffusion results in the original image. 

Here's my test:

#### Original
![alt text](/images/original.png)

#### Forward diffusion
![alt text](/images/ddim_forward.png)


#### Backward diffusion
![alt text](/images/ddim_backward.png)
