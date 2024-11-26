# Comprehensive PyTorch Cheat Sheet

![Abdullah Mirza AI and Machine Learning Developer](https://media.licdn.com/dms/image/v2/D4D16AQEOrUjAALRlBw/profile-displaybackgroundimage-shrink_350_1400/profile-displaybackgroundimage-shrink_350_1400/0/1728131435697?e=1736985600&v=beta&t=qXMjqAkCNagH3IYRfixQ6znlZAZ4P-qXsV6Bf16Jr28)


## Table of Contents

1. [PyTorch Basics](#pytorch-basics)
2. [Advanced Data Handling](#advanced-data-handling)
3. [Custom Training Loops](#custom-training-loops)
4. [Model Architectures](#model-architectures)
5. [Optimization and Training Tricks](#optimization-and-training-tricks)
6. [Advanced NLP](#advanced-nlp)
7. [Generative AI](#generative-ai)
8. [PyTorch Ecosystem](#pytorch-ecosystem)
   - [PyTorch Lightning](#pytorch-lightning)
   - [Hugging Face Transformers](#hugging-face-transformers)
   - [ONNX Runtime](#onnx-runtime)
   - [PyTorch Quantization](#pytorch-quantization)
   - [PyTorch Profiler](#pytorch-profiler)
9. [Model Deployment](#model-deployment)
10. [Debugging and Profiling](#debugging-and-profiling)
11. [Learning Resources](#learning-resources)
12. [General Additions](#general-additions)
    - [Model Interpretability](#model-interpretability)
    - [Ethical AI](#ethical-ai)
    - [Security Considerations](#security-considerations)

## PyTorch Basics

### Key Topics

- **Tensors:**
  - **Creation:** `torch.tensor`, `torch.zeros`, `torch.ones`, `torch.arange`, `torch.rand`, `torch.randn`
  - **Properties:** `.shape`, `.dtype`, `.device`
  - **Operations:** Basic math, matrix multiplication (`@` or `torch.matmul`), broadcasting
  - **Conversion:** NumPy ↔ PyTorch: `torch.from_numpy` and `.numpy()`

### Tips & Examples

```python
import torch

# Tensor creation and GPU acceleration
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.rand(3, 3, device=device)

# Basic operations
y = x + 2
z = x @ x.T  # Matrix multiplication

# Conversion to and from NumPy
x_np = x.cpu().numpy()
x_torch = torch.from_numpy(x_np).to(device)
```

## Advanced Data Handling

### Key Topics

- **Custom Datasets:**
  - Subclassing `torch.utils.data.Dataset`
  - Overriding `__len__` and `__getitem__`
- **DataLoader:**
  - Batch size, shuffling, `num_workers`
  - Prefetching and memory pinning
- **Transforms:**
  - `torchvision.transforms` for image augmentation
  - Custom transform functions

### Tips & Examples

```python
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.labels[idx]

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Create dataset and dataloader
dataset = CustomDataset(data, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2)
```

## Custom Training Loops

### Key Topics

- Manual forward-backward passes with `torch.autograd`
- Training workflow:
  - Model initialization
  - Loss calculation
  - Optimizer step and learning rate scheduling
- Gradient clipping for stabilization

### Tips & Examples

```python
import torch.optim as optim

# Define model, loss, and optimizer
model = CustomModel().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
```

## Model Architectures

### Key Topics

- **Sequential API:** Quick prototyping
- **Subclassing `torch.nn.Module`:** Advanced custom models
- **Building blocks:**
  - Fully connected (`Linear`), Convolutional (`Conv2d`), Recurrent (`LSTM`, `GRU`), Attention
  - Dropout, BatchNorm, and Activation Functions
- **Weight Initialization:** `torch.nn.init`

### Example

```python
import torch.nn as nn
import torch.nn.functional as F

class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Initialize model
model = CustomModel(input_size=784, hidden_size=128, output_size=10).to(device)
```

## Optimization and Training Tricks

### Key Topics

- **Optimizers:** SGD, Adam, AdamW
- **Schedulers:** Learning rate decay with `torch.optim.lr_scheduler`
- **Regularization:** Weight decay, dropout, and early stopping
- **Mixed Precision Training:** `torch.cuda.amp`
- **Distributed Training:**
  - `torch.nn.DataParallel`
  - `torch.distributed`

### Example: Mixed Precision Training

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for inputs, targets in dataloader:
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Advanced NLP

### Key Topics

- Tokenization (`torchtext`, Hugging Face `transformers`)
- Pre-trained Embeddings (GloVe, FastText)
- Sequence models (LSTM, GRU, Transformer)
- Attention Mechanisms and BERT-like architectures
- Hugging Face integration for fine-tuning

### Example: Using Pre-trained BERT

```python
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

inputs = tokenizer("This is a test", return_tensors="pt")
outputs = model(**inputs)
```

## Generative AI

### Key Topics

- **GANs:**
  - Generator and Discriminator design
  - Training stability: Wasserstein Loss, Gradient Penalty
- **Variational Autoencoders:**
  - Latent space representation
- **Transformers for Generation:**
  - GPT, BERT, T5
- **Diffusion Models** (e.g., DALL-E)

### Example: Simple GAN

```python
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

# Initialize models
generator = Generator(input_size=100, hidden_size=128, output_size=784).to(device)
discriminator = Discriminator(input_size=784, hidden_size=128, output_size=1).to(device)

# Training loop
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)
criterion = nn.BCELoss()

for epoch in range(num_epochs):
    for real_images, _ in dataloader:
        real_images = real_images.view(-1, 784).to(device)
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        outputs = discriminator(real_images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        z = torch.randn(batch_size, 100).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        z = torch.randn(batch_size, 100).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images)

        g_loss = criterion(outputs, real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f'Epoch {epoch+1}/{num_epochs}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')
```

## PyTorch Ecosystem

### Tools

- **torchvision:** Pre-trained models, datasets, transforms
- **torchaudio:** Audio processing
- **torchtext:** NLP utilities
- **PyTorch Lightning:** High-level training
- **Hugging Face Transformers**
- **FastAI:** Quick prototyping

### PyTorch Lightning

#### More Detailed Examples and Benefits

PyTorch Lightning simplifies the training loop by abstracting away the boilerplate code. It provides a high-level interface for PyTorch, making it easier to manage experiments and scale to multiple GPUs.

#### Example

```python
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

class LitModel(pl.LightningModule):
    def __init__(self):
        super(LitModel, self).__init__()
        self.layer = nn.Linear(32, 2)

    def forward(self, x):
        return torch.relu(self.layer(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

model = LitModel()
trainer = pl.Trainer()
trainer.fit(model)
```

### Hugging Face Transformers

#### More Examples on Fine-tuning and Using Different Pre-trained Models

Hugging Face Transformers provides a wide range of pre-trained models and tools for fine-tuning. It simplifies the process of working with state-of-the-art NLP models.

#### Example

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```

### ONNX Runtime

#### Information on Using ONNX Runtime for Optimized Inference

ONNX Runtime is a cross-platform, high-performance scoring engine for Open Neural Network Exchange (ONNX) models. It enables optimized inference on a variety of hardware platforms.

#### Example

```python
import onnxruntime as ort

ort_session = ort.InferenceSession("model.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)
```

### PyTorch Quantization

#### More Details on Quantization Techniques for Model Optimization

Quantization is a technique to reduce the model size and improve inference speed by reducing the precision of the model weights and activations.

#### Example

```python
model_fp32 = torch.quantization.QuantWrapper(model)
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model_fp32, inplace=True)
torch.quantization.convert(model_fp32, inplace=True)
```

### PyTorch Profiler

#### More Detailed Examples on Using the Profiler for Performance Analysis

The PyTorch Profiler provides detailed performance analysis of your model, helping you identify bottlenecks and optimize your code.

#### Example

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as p:
    for step, (images, labels) in enumerate(dataloader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

## Model Deployment

### Key Topics

- **TorchScript:** Exporting PyTorch models
- **ONNX:** Conversion for cross-platform inference
- **PyTorch Serve:** Deploying as REST API
- **Mobile and Edge Deployment:** Quantization and Pruning

### Example: Exporting a Model with TorchScript

```python
# Export the model
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")
```

### Example: Converting to ONNX

```python
import torch.onnx

# Convert the model
dummy_input = torch.randn(1, 3, 224, 224, device=device)
torch.onnx.export(model, dummy_input, "model.onnx")
```

## Debugging and Profiling

### Tools

- **torch.profiler:** Analyzing performance
- **TensorBoard:** Integration with PyTorch
- **Debugging gradients:** Check `.grad` values

### Example: Using TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

for n_iter, (images, labels) in enumerate(dataloader):
    writer.add_images('images', images, n_iter)
    writer.add_scalar('Loss/train', loss.item(), n_iter)
```

## Learning Resources

### Books

- _Deep Learning with PyTorch_ by Eli Stevens, Luca Antiga, and Thomas Viehmann
- _Programming PyTorch for Deep Learning_ by Ian Pointer

### Courses

- [FastAI](https://course.fast.ai/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

### Blogs and Repos

- [Papers with Code](https://paperswithcode.com/)
- [PyTorch Blog](https://pytorch.org/blog/)

## General Additions

### Model Interpretability

#### More Tools and Techniques for Model Interpretability

Model interpretability is crucial for understanding and trusting machine learning models. Techniques like SHAP and LIME help in explaining model predictions.

#### Example: Using SHAP

```python
import shap

explainer = shap.Explainer(model)
shap_values = explainer(x_test)
```

### Ethical AI

#### Guidelines and Tools for Ethical Considerations in AI

Ethical AI involves ensuring fairness, accountability, and transparency in AI systems. Tools like AIF360 help in mitigating bias and ensuring fairness.

#### Example: Using AIF360 for Bias Mitigation

```python
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset

dataset = BinaryLabelDataset(...)
rw = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
dataset_transf_train, dataset_transf_test = dataset.split([0.7], shuffle=True)
dataset_transf_train = rw.fit_transform(dataset_transf_train)
```

### Security Considerations

#### Best Practices for Securing Machine Learning Models and Data

Security in machine learning involves protecting models and data from adversarial attacks and ensuring privacy.

#### Example: Using Adversarial Robustness Toolbox

```python
from adversarial_robustness_toolbox.attacks import FGSM

attack = FGSM(model, eps=0.3)
x_adv = attack.generate(x_test)
```
