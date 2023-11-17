# DeepTHS

A framework to compute threshold sensitivity of deep networks to visual stimuli.

<span style="color:red">*I have converted this project into a pip package
called ["osculari"](https://pypi.org/project/osculari/)*. Please use the osculari package instead of
this repository.</span>.

This repository provides an easy interface to train a linear classifier on top of the extract
features
from pretrained networks implemented in PyTorch. It includes:

- Most models and pretrained networks
  from [PyTorch's official website](https://pytorch.org/vision/stable/models.html).
- [CLIP](https://github.com/openai/CLIP) language-vision model.
- [Taskonomy](http://taskonomy.stanford.edu/) networks.
- Different architectures (e.g., CNN and ViT).
- Different tasks (e.g., classification and segmentation).
- Training/testing routines for 2AFC and 4AFC tasks.

# Examples

## 2AFC Task

### Creating a model

Let's create a linear classifier to perform a binary-classification 2AFC
(two-alternative-force-choice) task. This is easily achieved by inheriting the
```readout.ClassifierNet``` .

``` python

from deepths.models import readout

class FeatureDiscrimination2AFC(readout.ClassifierNet):
    def __init__(self, classifier_kwargs, readout_kwargs):
        super(FeatureDiscrimination2AFC, self).__init__(
            input_nodes=2, num_classes=2, 
            **classifier_kwargs, **readout_kwargs
        )

    def forward(self, x0, x1):
        x0 = self.do_features(x0)
        x1 = self.do_features(x1)
        x = torch.cat([x0, x1], dim=1)
        return self.do_classifier(x)

```

The parameter ```input_nodes``` specify the number of images we are extracting
features from. The parameter ```num_classes``` specifies the number of outputs
for the linear classifier.

### Instantiating

Let's use ```ResNet50``` as our pretrained network and extract feature from
the layer ```area0```.

```python

net_name = 'resnet50'
weights = 'resnet50'
target_size = 224
readout_kwargs = {
    'architecture': net_name, 'target_size': target_size,
    'transfer_weights': [weights, 'area0'],

}
classifier_lwargs = {
    'classifier': 'nn',
    'pooling': None
}
net = FeatureDiscrimination2AFC(classifier_lwargs, readout_kwargs)

```

The variable ```readout_kwargs``` specifies the details of the *pretrained* network:

- ```architecture``` is network's architecture (e.g., ```ResNet50``` or ```ViT-B32```).
- ```transfer_weights``` defines the weights of the pretrained network and the layer(s) to use:
    * The first index must be either path to the pretrained weights or PyTorch supported weights (in
      this example we are using the default PyTorch weights of ```ResNet50```).
    * The second index is the read-out (cut-off) layer. In this example, we extract features
      from ```area0```.

The variable ```classifier_lwargs``` specifies the details of the *linear classifier*:

- ```classifier``` specifies types of linear classifier. It mainly supports neural network (*NN*)
  with partial support for SVM.
- ```pooling``` specifies whether to perform pooling over extracted features (without any new
  weights to learn). This is useful to reduce the dimensionality of the extracted features.

Let's print our network:

```
print(net)

FeatureDiscrimination2AFC(
  (backbone): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  )
  (fc): Linear(in_features=401408, out_features=2, bias=True)
)
```

We can see that it contains of two nodes: *backbone* and *fc* corresponding to the
pretrained network and linear classifier, respectively.

### Pooling

From the print above, we can observe that the dimensionality of the input to the
linear classifier is too large (a vector of 401408 elements). It might be of interest
to reduce this by means of pooling operations. For instance, we can make an instance
of ```FeatureDiscrimination2AFC``` with ```'pooling': 'avg_2_2'``` (i.e., average pooling over a
2-by-2 window).
In the new instance the input to the linear layer is only 512 elements.

```
FeatureDiscrimination2AFC(
  (backbone): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  )
  (pool_avg): AdaptiveAvgPool2d(output_size=(2, 2))
  (fc): Linear(in_features=512, out_features=2, bias=True)
)
```

- To use max pooling: ```'pooling': 'max_2_2'```.
- To pool over a different window: e.g., ```'pooling': 'max_5_3'``` pools over a 5-by-3 window.

# Script

### Contrast Sensitivity Function (CSF)

We have used this repository to measure
the [networks' CSF](https://arashakbarinia.github.io/projects/deepcsf/).

To train a contrast discriminator linear classifier:

```shell

python src/csf_train.py -aname $MODEL --transfer_weights $WEIGHTS $LAYER  \
  --target_size 224 --classifier "nn" \ 
  -dname $DB --data_dir $DATA_DIR --train_samples 15000 --val_sample 1000 \
  --contrast_space "rgb" --colour_space "imagenet_rgb" --vision_type "trichromat" \ 
  -b 64 --experiment_name $EXPERIMENT_NAME --output_dir $OUT_DIR  \
  -j 4 --gpu 0 --epochs 10 

```

To measure the CSFs (```$CONTRAST_SPACE``` can be one of the following values
"lum_ycc" "rg_ycc" "yb_ycc" corresponding to luminance, red-green and yellow-blue channels).

```shell
python csf_test.py -aname $MODEL_PATH --contrast_space $CONTRAST_SPACE  \
  --target_size 224 --classifier "nn" --mask_image "fixed_cycle"  \
  --experiment_name $EXPERIMENT_NAME  \
  --colour_space "imagenet_rgb"  --vision_type "trichromat"  \
  --print_freq 1000 --output_dir $OUT_DIR --gpu 0
```