# Architectural and Implementation Justification

This contains the documentation for the GRU gate bias ablation study using the s-CIFAR-10 dataset.

## 1. Introduction \& Formulation

The objective of the study is to evaluate the role of the reset $r_t$ and update $z_t$ gate biases in a GRU when processing long temporal sequences. To isolate long-range memory capacity, we use the s-CIFAR-10 benchmark, flattening $32\times32$ grids into 1024-step temporal sequences.

## 2. Data Pipeline (`src/data.py`)

Our data pipeline applies four transformations:

```python
transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        FlattenSequence()
    ])
```

### Sequential Grayscale Flattening

CIFAR-10 is regularly a spatial task, but reducing it to 1 channel forces the model to rely entirely on its hidden state $h_t$ to integrate information over 1024 time steps. In order to be able to classify the image at $t=1024$, the network must preserve feature representations from the entire image, all the way from $t=1$, which creates a stress test for the GRU's ability to preserve the history. Therefore, we create a stress test for the RNN's **B**ack**p**rop **T**hrough **T**ime (BPTT) against gradient problems. We implement a custom `class FlattenSequence()` for this transformation to ensure cross-compatability with Apple Silicon and Linux HPC clusters.

#### A note on BPTT

Conceptually, PyTorch unrolls the RNN into a 1024-layer deep feedforward network, where layer 1 corresponds to $t=1$, and layer $k$ with $t=k$, etc. BPTT is fundamentally the same algorithm as regular backprop, applied to this unrolled computer graph. The main difference, though, is in the weight tying; i.e., the $W_{hh}$ weight matrix is the same across the 1024 "layers." When calculating the gradient of the loss (at $t=1024$) w.r.t. the hidden state $h_{t=1}$, we must multiply the gradient by $W_{hh}$ 1024 times. To keep the gradients from exploding or vanishing, we therefore require orthogonal initialization (see below).

### Standardization

Inputs are normalized to have both a mean and standard deviation of 0.5. This ensures that our input is strictly bound between [-1,1]. The logic follows:

* We have raw data (CIFAR-10 image) with integer pixel values $\in [0,255]$.
* The `.ToTensor()` transform converts the PIL image to a PyTorch `tensor` and then scales the values (dividing by 255) to make the domain $[0.0, +1.0]$.
* The `.Normalize((0.5,), (0.5,))` transform then gives us a mean and standard deviation both of 0.5.
    * Therefore, the min possible value is $\frac{x_{\text{min}} - \mu}{\sigma} = \frac{0.0 - 0.5}{0.5} = -1.0$. 
    * The max possible value is $\frac{x_{\text{max}} - \mu}{\sigma} = \frac{1.0 - 0.5}{0.5} = +1.0$. 

Therefore, this means that our input domain is shifted into $[-1.0, +1.0]$.

Unscaled inputs in cont-time dynamical systems / deep RNNs can saturate the non-linearities (sigmoid/tanh) of the gates, which leads to zero-gradients. By bounding the inputs in $[-1.0, +1.0]$ we ensure that the initial linear projections $W_{ih}x_t$ remain within the linear regime of the sigmoid $\sigma(\cdot)$ and $\tanh(\cdot)$ activation functions during early stages of training.

## 3. Architectural Constraints and Mathematics (`src/model.py` and `src/utils.py`)

### Parameter Budgeting

We need to build a model of 100k parameters. We have a single-layer GRU with a hidden dim $H=181$. Following the math...

* Input size $X=1$
* Classes $C=10$
* GRU Params = $3H(X+H+2) = 3(181)(1+181+2)=99,912$
* Linear Params = $C(H+1) = 10(181+1)=1,820$
* Total Params = $101,732$.

The `utils.count_parameters()` function programatically verifies that we have the correct number of params.

### Orthogonal Initialization

The hidden state $h_t$ is repeatedly multiplied by $W_{hh}$ over the 1024 steps. If the eigenvalues of $W_{hh}$ are $>1$, then gradients explode, and conversely they vanish if the eigenvalues are $<1$. Therefore, we initialize $W_{hh}$ to be orthogonal, i.e., with eigenvalues exactly 1; thus, gradient magnitudes are preserved.

### Xavier Initialization

If the values of our weights are too large, passing a signal $x$ through a linear layer $y=Wx$ causes the variance of $y$ to become massive. We therefore risk passing huge values into our activation functions (`tanh` or `sigmoid`), which pushes them to their asymptotes (either -1, 0, or 1). The effect of this is that the gradients of the activation functions at these asymptotes is 0, and so this saturation problem causes the gradients to immediately die.

We instead apply a Xavier initialization to our input-to-hidden weights $W_{ih}$, so that our single pixel input enters the GRU without saturating the gates. Xavier sets the weights by drawing from a distribution:

$$
w \sim \mathcal{U} \left( - \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, + \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}} \right),
$$

where $w$ is a shorthand for each individual entry to the weight matrix $W_{ih}$. The bound ensures that the variance of the input signal is preserved as it passes through the weights and into the hidden state.

### Baseline Bias Initialization

In our baseline model, the biases are learnable. Here, we explicitly initialize the update gate bias $b_z$ to `+1.0`, whilst keeping the reset $b_r$ and candidate $b_n$ biases initialized to `0.0`.

Recall the following four equations which describe the GRU, per the [PyTorch docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html):

$$
\begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) \odot n_t + z_t \odot h_{(t-1)}.
\end{array}
$$

Recall the definition of the sigmoid function:

$$
\sigma(z) = \frac{1}{1+e^{-z}}.
$$

Now, at step $t=1$, immediately after initialization, the individual weights of $W_{iz}$ and $W_{hz}$ are very small random numbers, and $h_{t-1}$ defaults to a tensor of zeros in PyTorch. The update gate equation then simplifies to:

$$
z_t \approx \sigma(b_z) \text{ when $t$ small, e.g., when $t-1$.}
$$

* To remember the past state $h_{t-1}$, we need $z_t \to 1$. 
* An initialization of $b_z=0.0$ gives $z_t \approx 1/(1+e^0) = 0.5$. A 50\% decay over 1024 steps causes catastrophic forgetting.
* An initialization of $b_z=1.0$ gives $z_t \approx 1/(1+e^{-1}) \approx 0.731$. Therefore, we keep \~73\% of the old memory and mathematically bias the network to remember. This chrono-initialization lengthens the time-constant of the memory gates, which allows gradients to flow back to $t=1$ in the first epoch, and the net can then learn to fine-tune this bias.

*Note on Chrono-Initialization:* We empirically tested true Chrono-Initialization ($`b_z\sim\log(\mathcal{U}[1, 1024])`$) as proposed in the literature. However, this yielded catastrophic gradient saturation. Because the GRU update gate dictates both memory and input weighting simultaneously ($z_t$ and $1-z_t$), biases averaging $+6.2$ caused the network to permanently ignore new inputs ($z_t=\sigma(6.2) \approx 0.998$). The $+1.0$ uniform initialization is a better choice for preserving BPTT gradients while maintaining input sensitivity.

## 4. Ablation Implementation (`src/model.py`)

We need to freeze at zero the reset $b_r$ and update $b_z$ biases, allowing the candidate state bias to remain learnable. PyTorch fuses all three biases $\{b_r, b_u, b_n\}$ into (for layer 0) the input-hidden biases tensor `bias_ih_l0` and the hidden-hidden biases tensor `bias_hh_l0`. With a hidden dimension $H=181$, these have length of $3H=543$. PyTorch concatenates three biases into the vectors:

* Indices `[0:H]` are the reset gate
* Indices `[H:2H]` are the update gate
* Indices `[2H:3H]` are the candidate gate.

```python
def _initialize_weights(self):
        ...
            elif 'bias' in name:
                param.data.fill_(0.)

                if not self.ablate_biases:
                    # 
                    update_bias_start = self.hidden_size
                    update_bias_end = 2 * self.hidden_size
                    param.data[update_bias_start:update_bias_end].fill_(1.)
        
        if self.ablate_biases:
            self._apply_bias_ablation_hooks()
```

Our coding logic does this:

1. First, we do `param.data.fill_(0.0)`. This sets the entire vector of length 543 to `0.0`, i.e., all three gates are init'd to `0.0`.
2. Then, if we are in the baseline model, we overwrite the middle chunk to `1.0` through the selection statement and `param.data[update_bias_start:update_bias_end].fill_(1.)`.
3. However, if we are in the ablated model, we ignore this selection statement and so all 543 entries are left to `0.0`. We need, though, to **ensure that the reset and update gates _stay_ at `0.0`**, and therefore we implement a backward hook (see below). 


### Native PyTorch vs. Custom Loop

We have two options to go ahead and freeze a portion of the tensor:

1. Write a custom GRU cell in Python using a `for` loop to iterate over 1024 steps.
2. Use PyTorch's `nn.GRU` and apply a hook.

Clearly option 2 is the better option because we preserve the ability to use the fused cuDNN kernels that `torch` natively has, which are required for training at speed.

### The Hook Mechanism

Inside the `class sCIFAR10_GRU`, we have the following function that applies a hook:

```python
def _apply_bias_ablation_hooks(self):
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                def hook(grad):
                    grad_clone = grad.clone()
                    grad_clone[:2*self.hidden_size] = 0.
                    return grad_clone
                param.register_hook(hook)
```

This intercepts the **gradient** tensor (of length 543) and takes the first $2H=362$ elements which correspond to the update and reset gates, and sets them to 0. This means that, when the optimizer steps, it does `weight -= lr * grad`, and with `grad=0` for these specific entries, we ensure that the weight remains at `0.0`, as initialised. In this way, the candidate bias weights _are_ updated during training, as desired, because we leave their gradients untouched.

## 5. Training Dynamics (`src/train.py` and `src/run.py`)

### Optimizer

We use the `AdamW` optimizer over standard `Adam`. The latter applies L2 regularization by adding the weights to the raw gradients _before_ calculating the moving averages (i.e., momentum). In our case, the BPTT gradients are already sensitive and chaotic, and mixing weight decay directly into these can corrupt momentum estimators. `AdamW` instead decouples the weight decay and only applies it _after_ the adaptive gradient update, which ensures that the momentum in fact tracks the loss landscape. We need this for stability.

### Learning Rate

Originally, we implemented a base learning rate of `1e-3` but found this to be too high, causing spiking and lack of convergence. Instead, we improved this to a base learning rate of `5e-4` with a `CosineAnnealingLR` scheduler, reducing the learning rate to `1e-5` over 40 epochs to promote convergence through the loss landscape. The training was performed over 40 epochs to ensure convergence, and although the [training loss graph](report/figures/training_loss_comparison.png) indicates that further training could have slightly improved convergence and final model performance, ensuring maximum performance was not the purpose of this task.

### Batch Size

We selected a batch size of 128 primarily for optimization dynamics. Since we used an RTX 5000 to train, OOM errors are not a concern given the small model size (100K params) that we use. We avoided using a larger batch size because, although this could yield a more accurate gradient estimate, it also can mean we are (1) prone to overfitting (because it doesn't generalize as well to unseen data) and, similarly, that training does not converge into a global minimum but rather gets stuck in a local minimum. A smaller batch size, on the other hand, would introduce too much variance and not allow the model to learn properly, given our need to remember a longer memory. A batch size of 128 strikes a balance between these two, providing sufficient stochastic noise in the gradients to escape local minima, whilst remaining stable to benefit from the momentum of the optimizer.