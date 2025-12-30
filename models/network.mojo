"""
3D Convolutional Neural Network for MRI Classification
=======================================================
Implements VGG-style and ResNet-style architectures for medical imaging.
"""

from .tensor3d import Tensor3D, Shape5D, FloatDType
from .layers import Conv3D, BatchNorm3D, MaxPool3D, GlobalAvgPool3D, Linear, Dropout3D
from .activations import ReLU, Softmax


struct ConvBlock3D:
    """
    Convolutional block: Conv3D -> BatchNorm3D -> ReLU
    """
    var conv: Conv3D
    var bn: BatchNorm3D
    var relu: ReLU
    
    fn __init__(inout self, in_channels: Int, out_channels: Int, kernel_size: Int = 3, padding: Int = 1):
        self.conv = Conv3D(in_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn = BatchNorm3D(out_channels)
        self.relu = ReLU()
    
    fn forward(inout self, x: Tensor3D) -> Tensor3D:
        var out = self.conv.forward(x)
        out = self.bn.forward(out)
        out = self.relu.forward(out)
        return out
    
    fn backward(inout self, grad: Tensor3D) -> Tensor3D:
        var g = self.relu.backward(grad)
        g = self.bn.backward(g)
        g = self.conv.backward(g)
        return g
    
    fn train(inout self):
        self.bn.train()
    
    fn eval(inout self):
        self.bn.eval()


struct VGG3D:
    """
    VGG-style 3D CNN for MRI classification.
    
    Architecture:
    - Block 1: Conv(1->32) -> BN -> ReLU -> MaxPool
    - Block 2: Conv(32->64) -> BN -> ReLU -> MaxPool
    - Block 3: Conv(64->128) -> BN -> ReLU -> MaxPool
    - Block 4: Conv(128->256) -> BN -> ReLU -> GlobalAvgPool
    - Classifier: Linear(256->128) -> ReLU -> Dropout -> Linear(128->num_classes)
    """
    var in_channels: Int
    var num_classes: Int
    var base_filters: Int
    
    # Convolutional blocks
    var block1: ConvBlock3D
    var pool1: MaxPool3D
    var block2: ConvBlock3D
    var pool2: MaxPool3D
    var block3: ConvBlock3D
    var pool3: MaxPool3D
    var block4: ConvBlock3D
    var global_pool: GlobalAvgPool3D
    
    # Classifier
    var fc1: Linear
    var fc1_relu: ReLU
    var dropout: Dropout3D
    var fc2: Linear
    var softmax: Softmax
    
    var training: Bool
    
    fn __init__(
        inout self,
        in_channels: Int = 1,
        num_classes: Int = 2,
        base_filters: Int = 32,
        dropout_rate: Float32 = 0.5
    ):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_filters = base_filters
        self.training = True
        
        # Build convolutional blocks
        self.block1 = ConvBlock3D(in_channels, base_filters)
        self.pool1 = MaxPool3D(kernel_size=2, stride=2)
        
        self.block2 = ConvBlock3D(base_filters, base_filters * 2)
        self.pool2 = MaxPool3D(kernel_size=2, stride=2)
        
        self.block3 = ConvBlock3D(base_filters * 2, base_filters * 4)
        self.pool3 = MaxPool3D(kernel_size=2, stride=2)
        
        self.block4 = ConvBlock3D(base_filters * 4, base_filters * 8)
        self.global_pool = GlobalAvgPool3D()
        
        # Classifier
        let hidden_features = 128
        self.fc1 = Linear(base_filters * 8, hidden_features)
        self.fc1_relu = ReLU()
        self.dropout = Dropout3D(dropout_rate)
        self.fc2 = Linear(hidden_features, num_classes)
        self.softmax = Softmax()
    
    fn forward(inout self, x: Tensor3D) -> Tensor3D:
        """
        Forward pass through the network.
        
        Input: (batch, 1, D, H, W) - MRI volume
        Output: (batch, num_classes, 1, 1, 1) - class probabilities
        """
        # Convolutional feature extraction
        var out = self.block1.forward(x)
        out = self.pool1.forward(out)
        
        out = self.block2.forward(out)
        out = self.pool2.forward(out)
        
        out = self.block3.forward(out)
        out = self.pool3.forward(out)
        
        out = self.block4.forward(out)
        out = self.global_pool.forward(out)
        
        # Flatten for classifier (already (batch, channels, 1, 1, 1))
        out = self.fc1.forward(out)
        out = self.fc1_relu.forward(out)
        out = self.dropout.forward(out)
        out = self.fc2.forward(out)
        out = self.softmax.forward(out)
        
        return out
    
    fn backward(inout self, grad: Tensor3D) -> Tensor3D:
        """Backward pass through the network."""
        var g = self.softmax.backward(grad)
        g = self.fc2.backward(g)
        g = self.dropout.backward(g)
        g = self.fc1_relu.backward(g)
        g = self.fc1.backward(g)
        
        g = self.global_pool.backward(g)
        g = self.block4.backward(g)
        
        g = self.pool3.backward(g)
        g = self.block3.backward(g)
        
        g = self.pool2.backward(g)
        g = self.block2.backward(g)
        
        g = self.pool1.backward(g)
        g = self.block1.backward(g)
        
        return g
    
    fn parameters(self) -> List[Tensor3D]:
        """Get all trainable parameters."""
        var params = List[Tensor3D]()
        
        # Block 1
        for p in self.block1.conv.parameters():
            params.append(p[])
        params.append(self.block1.bn.gamma)
        params.append(self.block1.bn.beta)
        
        # Block 2
        for p in self.block2.conv.parameters():
            params.append(p[])
        params.append(self.block2.bn.gamma)
        params.append(self.block2.bn.beta)
        
        # Block 3
        for p in self.block3.conv.parameters():
            params.append(p[])
        params.append(self.block3.bn.gamma)
        params.append(self.block3.bn.beta)
        
        # Block 4
        for p in self.block4.conv.parameters():
            params.append(p[])
        params.append(self.block4.bn.gamma)
        params.append(self.block4.bn.beta)
        
        # Classifier
        params.append(self.fc1.weight)
        params.append(self.fc1.bias)
        params.append(self.fc2.weight)
        params.append(self.fc2.bias)
        
        return params
    
    fn gradients(self) -> List[Tensor3D]:
        """Get all parameter gradients."""
        var grads = List[Tensor3D]()
        
        # Block 1
        for g in self.block1.conv.gradients():
            grads.append(g[])
        grads.append(self.block1.bn.gamma_grad)
        grads.append(self.block1.bn.beta_grad)
        
        # Block 2
        for g in self.block2.conv.gradients():
            grads.append(g[])
        grads.append(self.block2.bn.gamma_grad)
        grads.append(self.block2.bn.beta_grad)
        
        # Block 3
        for g in self.block3.conv.gradients():
            grads.append(g[])
        grads.append(self.block3.bn.gamma_grad)
        grads.append(self.block3.bn.beta_grad)
        
        # Block 4
        for g in self.block4.conv.gradients():
            grads.append(g[])
        grads.append(self.block4.bn.gamma_grad)
        grads.append(self.block4.bn.beta_grad)
        
        # Classifier
        grads.append(self.fc1.weight_grad)
        grads.append(self.fc1.bias_grad)
        grads.append(self.fc2.weight_grad)
        grads.append(self.fc2.bias_grad)
        
        return grads
    
    fn train(inout self):
        """Set model to training mode."""
        self.training = True
        self.block1.train()
        self.block2.train()
        self.block3.train()
        self.block4.train()
        self.dropout.train()
    
    fn eval(inout self):
        """Set model to evaluation mode."""
        self.training = False
        self.block1.eval()
        self.block2.eval()
        self.block3.eval()
        self.block4.eval()
        self.dropout.eval()
    
    fn num_parameters(self) -> Int:
        """Count total trainable parameters."""
        var total = 0
        for p in self.parameters():
            total += p[].numel()
        return total


# ============ ResNet-style Architecture ============

struct ResidualBlock3D:
    """
    3D Residual block with skip connection.
    """
    var conv1: Conv3D
    var bn1: BatchNorm3D
    var relu1: ReLU
    var conv2: Conv3D
    var bn2: BatchNorm3D
    var relu2: ReLU
    
    # Downsample for dimension matching
    var downsample: Bool
    var downsample_conv: Conv3D
    var downsample_bn: BatchNorm3D
    
    var _skip_cache: Tensor3D
    
    fn __init__(inout self, in_channels: Int, out_channels: Int, stride: Int = 1):
        self.conv1 = Conv3D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm3D(out_channels)
        self.relu1 = ReLU()
        self.conv2 = Conv3D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm3D(out_channels)
        self.relu2 = ReLU()
        
        # Downsample if dimensions change
        self.downsample = (in_channels != out_channels) or (stride != 1)
        self.downsample_conv = Conv3D(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.downsample_bn = BatchNorm3D(out_channels)
        
        self._skip_cache = Tensor3D.zeros(Shape5D(1, 1, 1, 1, 1))
    
    fn forward(inout self, x: Tensor3D) -> Tensor3D:
        """Forward with residual connection."""
        # Store skip for backward
        if self.downsample:
            var skip = self.downsample_conv.forward(x)
            self._skip_cache = self.downsample_bn.forward(skip)
        else:
            self._skip_cache = x
        
        # Main path
        var out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        
        # Add residual
        out = out.add(self._skip_cache)
        out = self.relu2.forward(out)
        
        return out
    
    fn backward(inout self, grad: Tensor3D) -> Tensor3D:
        """Backward pass with skip connection."""
        var g = self.relu2.backward(grad)
        
        # Gradient splits into main path and skip path
        var g_main = self.bn2.backward(g)
        g_main = self.conv2.backward(g_main)
        g_main = self.relu1.backward(g_main)
        g_main = self.bn1.backward(g_main)
        g_main = self.conv1.backward(g_main)
        
        # Skip path gradient
        var g_skip: Tensor3D
        if self.downsample:
            g_skip = self.downsample_bn.backward(g)
            g_skip = self.downsample_conv.backward(g_skip)
        else:
            g_skip = g
        
        # Combine gradients
        return g_main.add(g_skip)


struct ResNet3D:
    """
    ResNet-style 3D CNN for MRI classification.
    
    Architecture:
    - Initial: Conv(1->32) -> BN -> ReLU -> MaxPool
    - Layer 1: 2x ResBlock(32->32)
    - Layer 2: 2x ResBlock(32->64, stride=2)
    - Layer 3: 2x ResBlock(64->128, stride=2)
    - GlobalAvgPool -> Linear(128->num_classes)
    """
    var num_classes: Int
    var base_filters: Int
    
    # Initial layers
    var conv1: Conv3D
    var bn1: BatchNorm3D
    var relu1: ReLU
    var pool1: MaxPool3D
    
    # Residual layers
    var layer1_block1: ResidualBlock3D
    var layer1_block2: ResidualBlock3D
    var layer2_block1: ResidualBlock3D
    var layer2_block2: ResidualBlock3D
    var layer3_block1: ResidualBlock3D
    var layer3_block2: ResidualBlock3D
    
    # Classifier
    var global_pool: GlobalAvgPool3D
    var fc: Linear
    var softmax: Softmax
    
    fn __init__(inout self, in_channels: Int = 1, num_classes: Int = 2, base_filters: Int = 32):
        self.num_classes = num_classes
        self.base_filters = base_filters
        
        # Initial conv
        self.conv1 = Conv3D(in_channels, base_filters, kernel_size=7, stride=2, padding=3)
        self.bn1 = BatchNorm3D(base_filters)
        self.relu1 = ReLU()
        self.pool1 = MaxPool3D(kernel_size=2, stride=2)
        
        # Residual layers
        self.layer1_block1 = ResidualBlock3D(base_filters, base_filters)
        self.layer1_block2 = ResidualBlock3D(base_filters, base_filters)
        
        self.layer2_block1 = ResidualBlock3D(base_filters, base_filters * 2, stride=2)
        self.layer2_block2 = ResidualBlock3D(base_filters * 2, base_filters * 2)
        
        self.layer3_block1 = ResidualBlock3D(base_filters * 2, base_filters * 4, stride=2)
        self.layer3_block2 = ResidualBlock3D(base_filters * 4, base_filters * 4)
        
        # Classifier
        self.global_pool = GlobalAvgPool3D()
        self.fc = Linear(base_filters * 4, num_classes)
        self.softmax = Softmax()
    
    fn forward(inout self, x: Tensor3D) -> Tensor3D:
        """Forward pass."""
        var out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)
        
        out = self.layer1_block1.forward(out)
        out = self.layer1_block2.forward(out)
        
        out = self.layer2_block1.forward(out)
        out = self.layer2_block2.forward(out)
        
        out = self.layer3_block1.forward(out)
        out = self.layer3_block2.forward(out)
        
        out = self.global_pool.forward(out)
        out = self.fc.forward(out)
        out = self.softmax.forward(out)
        
        return out
    
    fn backward(inout self, grad: Tensor3D) -> Tensor3D:
        """Backward pass."""
        var g = self.softmax.backward(grad)
        g = self.fc.backward(g)
        g = self.global_pool.backward(g)
        
        g = self.layer3_block2.backward(g)
        g = self.layer3_block1.backward(g)
        
        g = self.layer2_block2.backward(g)
        g = self.layer2_block1.backward(g)
        
        g = self.layer1_block2.backward(g)
        g = self.layer1_block1.backward(g)
        
        g = self.pool1.backward(g)
        g = self.relu1.backward(g)
        g = self.bn1.backward(g)
        g = self.conv1.backward(g)
        
        return g


# Factory function
fn create_model(
    model_type: String,
    in_channels: Int = 1,
    num_classes: Int = 2,
    base_filters: Int = 32
) -> VGG3D:
    """
    Create a 3D CNN model.
    
    Args:
        model_type: "vgg3d" or "resnet3d"
        in_channels: Number of input channels (1 for MRI)
        num_classes: Number of output classes
        base_filters: Base number of filters
    
    Returns:
        Configured model (VGG3D for now)
    """
    return VGG3D(in_channels, num_classes, base_filters)
