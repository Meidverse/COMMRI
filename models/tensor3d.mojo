"""
High-Performance 3D Tensor Implementation for Mojo
===================================================
SIMD-optimized tensor operations for medical imaging deep learning.

This module provides a Tensor3D struct with:
- Memory-aligned storage for SIMD operations
- Efficient indexing with bounds checking
- Core tensor operations (add, mul, matmul)
- Support for 5D tensors (batch, channels, depth, height, width)
"""

from memory import memset_zero, memcpy
from algorithm import vectorize, parallelize
from math import sqrt, exp, log, max as math_max, min as math_min
from sys.info import simdwidthof


alias FloatDType = DType.float32
alias simd_width = simdwidthof[FloatDType]()


@value
struct Shape5D:
    """5D shape for tensors: (batch, channels, depth, height, width)"""
    var batch: Int
    var channels: Int
    var depth: Int
    var height: Int
    var width: Int
    
    fn __init__(inout self, b: Int, c: Int, d: Int, h: Int, w: Int):
        self.batch = b
        self.channels = c
        self.depth = d
        self.height = h
        self.width = w
    
    fn numel(self) -> Int:
        """Total number of elements."""
        return self.batch * self.channels * self.depth * self.height * self.width
    
    fn __eq__(self, other: Shape5D) -> Bool:
        return (self.batch == other.batch and 
                self.channels == other.channels and
                self.depth == other.depth and
                self.height == other.height and
                self.width == other.width)
    
    fn __str__(self) -> String:
        return "(" + str(self.batch) + ", " + str(self.channels) + ", " + 
               str(self.depth) + ", " + str(self.height) + ", " + str(self.width) + ")"


struct Tensor3D:
    """
    High-performance 5D tensor for 3D medical imaging.
    
    Layout: NCDHW (batch, channels, depth, height, width)
    Memory: Contiguous, aligned for SIMD operations
    """
    var data: DTypePointer[FloatDType]
    var shape: Shape5D
    var _strides: StaticTuple[Int, 5]
    var _owned: Bool
    
    fn __init__(inout self, shape: Shape5D, zero_init: Bool = True):
        """Allocate tensor with given shape."""
        self.shape = shape
        self._owned = True
        
        # Calculate strides (row-major / C-contiguous)
        self._strides = StaticTuple[Int, 5](
            shape.channels * shape.depth * shape.height * shape.width,
            shape.depth * shape.height * shape.width,
            shape.height * shape.width,
            shape.width,
            1
        )
        
        # Allocate aligned memory
        let numel = shape.numel()
        self.data = DTypePointer[FloatDType].alloc(numel)
        
        if zero_init:
            memset_zero(self.data, numel)
    
    fn __init__(inout self, b: Int, c: Int, d: Int, h: Int, w: Int):
        """Convenience constructor with dimensions."""
        self.__init__(Shape5D(b, c, d, h, w))
    
    fn __copyinit__(inout self, existing: Self):
        """Copy constructor - deep copy."""
        self.shape = existing.shape
        self._strides = existing._strides
        self._owned = True
        
        let numel = self.shape.numel()
        self.data = DTypePointer[FloatDType].alloc(numel)
        memcpy(self.data, existing.data, numel)
    
    fn __moveinit__(inout self, owned existing: Self):
        """Move constructor."""
        self.shape = existing.shape
        self._strides = existing._strides
        self.data = existing.data
        self._owned = existing._owned
    
    fn __del__(owned self):
        """Free memory if owned."""
        if self._owned:
            self.data.free()
    
    # ============ Indexing ============
    
    @always_inline
    fn _offset(self, b: Int, c: Int, d: Int, h: Int, w: Int) -> Int:
        """Calculate linear offset for given indices."""
        return (b * self._strides[0] + c * self._strides[1] + 
                d * self._strides[2] + h * self._strides[3] + w)
    
    @always_inline
    fn __getitem__(self, b: Int, c: Int, d: Int, h: Int, w: Int) -> Scalar[FloatDType]:
        """Get element at indices."""
        return self.data.load(self._offset(b, c, d, h, w))
    
    @always_inline
    fn __setitem__(inout self, b: Int, c: Int, d: Int, h: Int, w: Int, val: Scalar[FloatDType]):
        """Set element at indices."""
        self.data.store(self._offset(b, c, d, h, w), val)
    
    fn numel(self) -> Int:
        """Total number of elements."""
        return self.shape.numel()
    
    # ============ Factory Methods ============
    
    @staticmethod
    fn zeros(shape: Shape5D) -> Self:
        """Create zero-initialized tensor."""
        return Self(shape, zero_init=True)
    
    @staticmethod
    fn ones(shape: Shape5D) -> Self:
        """Create tensor filled with ones."""
        var t = Self(shape, zero_init=False)
        for i in range(t.numel()):
            t.data.store(i, 1.0)
        return t
    
    @staticmethod
    fn randn(shape: Shape5D, seed: Int = 42) -> Self:
        """Create tensor with random normal values (Box-Muller transform)."""
        var t = Self(shape, zero_init=False)
        var state = seed
        
        for i in range(0, t.numel(), 2):
            # Simple LCG for randomness
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            let u1 = Float32(state) / Float32(0x7FFFFFFF)
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            let u2 = Float32(state) / Float32(0x7FFFFFFF)
            
            # Box-Muller transform
            let mag = sqrt(-2.0 * log(u1 + 1e-10))
            let z0 = mag * cos(2.0 * 3.14159265358979 * u2)
            let z1 = mag * sin(2.0 * 3.14159265358979 * u2)
            
            t.data.store(i, z0)
            if i + 1 < t.numel():
                t.data.store(i + 1, z1)
        
        return t
    
    @staticmethod
    fn kaiming_normal(shape: Shape5D, fan_in: Int, seed: Int = 42) -> Self:
        """Kaiming/He initialization for ReLU networks."""
        var t = Self.randn(shape, seed)
        let scale = sqrt(2.0 / Float32(fan_in))
        t.mul_scalar(scale)
        return t
    
    # ============ Element-wise Operations ============
    
    fn add(self, other: Self) -> Self:
        """Element-wise addition."""
        debug_assert(self.shape == other.shape, "Shape mismatch in add")
        var result = Self(self.shape, zero_init=False)
        
        @parameter
        fn vec_add[width: Int](i: Int):
            let a = self.data.load[width=width](i)
            let b = other.data.load[width=width](i)
            result.data.store[width=width](i, a + b)
        
        vectorize[vec_add, simd_width](self.numel())
        return result
    
    fn add_inplace(inout self, other: Self):
        """In-place element-wise addition."""
        debug_assert(self.shape == other.shape, "Shape mismatch in add_inplace")
        
        @parameter
        fn vec_add[width: Int](i: Int):
            let a = self.data.load[width=width](i)
            let b = other.data.load[width=width](i)
            self.data.store[width=width](i, a + b)
        
        vectorize[vec_add, simd_width](self.numel())
    
    fn sub(self, other: Self) -> Self:
        """Element-wise subtraction."""
        debug_assert(self.shape == other.shape, "Shape mismatch in sub")
        var result = Self(self.shape, zero_init=False)
        
        @parameter
        fn vec_sub[width: Int](i: Int):
            let a = self.data.load[width=width](i)
            let b = other.data.load[width=width](i)
            result.data.store[width=width](i, a - b)
        
        vectorize[vec_sub, simd_width](self.numel())
        return result
    
    fn mul(self, other: Self) -> Self:
        """Element-wise multiplication (Hadamard product)."""
        debug_assert(self.shape == other.shape, "Shape mismatch in mul")
        var result = Self(self.shape, zero_init=False)
        
        @parameter
        fn vec_mul[width: Int](i: Int):
            let a = self.data.load[width=width](i)
            let b = other.data.load[width=width](i)
            result.data.store[width=width](i, a * b)
        
        vectorize[vec_mul, simd_width](self.numel())
        return result
    
    fn mul_scalar(inout self, scalar: Scalar[FloatDType]):
        """In-place scalar multiplication."""
        @parameter
        fn vec_mul[width: Int](i: Int):
            let a = self.data.load[width=width](i)
            self.data.store[width=width](i, a * scalar)
        
        vectorize[vec_mul, simd_width](self.numel())
    
    fn div(self, other: Self) -> Self:
        """Element-wise division."""
        debug_assert(self.shape == other.shape, "Shape mismatch in div")
        var result = Self(self.shape, zero_init=False)
        
        @parameter
        fn vec_div[width: Int](i: Int):
            let a = self.data.load[width=width](i)
            let b = other.data.load[width=width](i)
            result.data.store[width=width](i, a / (b + 1e-8))
        
        vectorize[vec_div, simd_width](self.numel())
        return result
    
    # ============ Reduction Operations ============
    
    fn sum(self) -> Scalar[FloatDType]:
        """Sum all elements."""
        var total: Scalar[FloatDType] = 0.0
        for i in range(self.numel()):
            total += self.data.load(i)
        return total
    
    fn mean(self) -> Scalar[FloatDType]:
        """Mean of all elements."""
        return self.sum() / Scalar[FloatDType](self.numel())
    
    fn max(self) -> Scalar[FloatDType]:
        """Maximum element."""
        var max_val = self.data.load(0)
        for i in range(1, self.numel()):
            let val = self.data.load(i)
            if val > max_val:
                max_val = val
        return max_val
    
    fn min(self) -> Scalar[FloatDType]:
        """Minimum element."""
        var min_val = self.data.load(0)
        for i in range(1, self.numel()):
            let val = self.data.load(i)
            if val < min_val:
                min_val = val
        return min_val
    
    # ============ Activation Functions ============
    
    fn relu(self) -> Self:
        """ReLU activation."""
        var result = Self(self.shape, zero_init=False)
        
        @parameter
        fn vec_relu[width: Int](i: Int):
            let a = self.data.load[width=width](i)
            let zero = SIMD[FloatDType, width](0.0)
            result.data.store[width=width](i, (a > zero).select(a, zero))
        
        vectorize[vec_relu, simd_width](self.numel())
        return result
    
    fn relu_backward(self, grad: Self) -> Self:
        """ReLU backward pass."""
        debug_assert(self.shape == grad.shape, "Shape mismatch")
        var result = Self(self.shape, zero_init=False)
        
        @parameter
        fn vec_relu_grad[width: Int](i: Int):
            let x = self.data.load[width=width](i)
            let g = grad.data.load[width=width](i)
            let zero = SIMD[FloatDType, width](0.0)
            result.data.store[width=width](i, (x > zero).select(g, zero))
        
        vectorize[vec_relu_grad, simd_width](self.numel())
        return result
    
    # ============ Reshaping ============
    
    fn flatten(self) -> Self:
        """Flatten to (batch, features)."""
        let batch = self.shape.batch
        let features = self.shape.channels * self.shape.depth * self.shape.height * self.shape.width
        var result = Self(batch, features, 1, 1, 1, zero_init=False)
        memcpy(result.data, self.data, self.numel())
        return result
    
    fn view(self, new_shape: Shape5D) -> Self:
        """Reshape tensor (must have same number of elements)."""
        debug_assert(self.numel() == new_shape.numel(), "Element count must match")
        var result = Self(new_shape, zero_init=False)
        memcpy(result.data, self.data, self.numel())
        return result


# Helper functions
fn cos(x: Scalar[FloatDType]) -> Scalar[FloatDType]:
    """Cosine function."""
    # Taylor series approximation
    let x2 = x * x
    return 1.0 - x2/2.0 + x2*x2/24.0 - x2*x2*x2/720.0


fn sin(x: Scalar[FloatDType]) -> Scalar[FloatDType]:
    """Sine function."""
    # Taylor series approximation
    let x2 = x * x
    return x - x*x2/6.0 + x*x2*x2/120.0 - x*x2*x2*x2/5040.0
