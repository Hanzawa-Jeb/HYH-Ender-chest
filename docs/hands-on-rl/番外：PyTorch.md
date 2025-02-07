# 1. `Basics of Tensors`
- Tensors就是张量，用法与Numpy中的类似，支持并行运算
- 可以通过语句来将普通的多维列表或者numpy数组转化为pytorch tensor:
```python
#py_list case
data = [[1, 2], [3, 4]]
t_tensor = torch.tensor(data)
#numpy arrays case
np_array = np.array(data)
t_tensor2 = torch.from_numpy(np_array)
```
- 关于如何创建一个固定尺寸的随机张量，全一张量或者全零张量，***张量的尺寸可以用元组来进行指定， 一般会在结尾打上逗号***,  而且应该注意shape应该是先行再列。还有一个是
```python
shape = (2, 3, )
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

zeros_tensor_2 = torch.zeros(2, 3)
```
- 可以通过一系列命令来访问`torch.tensor`的特定属性， 例如***数据类型， 形状， 存储的设备***，注意`torch.tensor`与C中的数组类似，内部存储的数据类型都必须全部相同。
```python
tensor = torch.rand(3, 4)

print(f"Shape is {tensor.shape}")
print(f"datatype is {tensor.dtype}")
print(f"Device Tensor is on {tensor.device}")
```
- 一般我们会将***向量运算放到GPU上运行***，可以显式地将张量存储到GPU上
```python
if torch.cuda.is_available():
	tensor = tensor.to('cuda')
	print(f"device is {tensor.device}")
```
- 有关于`slicing`的使用：
```Python
tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(tensor[:, 2]) #意思是打出所有行的第二列
print(tensor[1:3, 0:2]) #意思是打出1-3行的0-2列
```
- 有关`concatenate`：如果dim = 1则是在行向进行拼接，即每行的长度都会边长，如果dim = 0则是在列向进行拼接，即每列的长度都会变长
```python
tensor = torch.zeros(2, 2)
new_tensor = torch.cat([tensor, tensor, tensor], dim = 1)
```
- 乘法（`element-wise form or matrix-vector form`）
```python
element_wise_op1 = tensor1.mul(tensor2)
element_wise_op2 = tensor1 * tensor2
element_wise_op1 == element_wise_op2

matrix_op1 = tensor1.matmul(tensor2)
matrix_op2 = tensor1 @ tensor2
```
- 转置`Transpose`
```python
tensor1 = tensor1.T
```
# 2. `torch.autograd`
- 神经网络就是一种复杂的***嵌套函数*** `(Nested Functions)`， 参数包含了`weights, biases`
- `Consisting of two steps: Forward Propagation and Backward Propagation`， 前一个负责计算出估计值，后一个负责通过链式规则来用结果调整参数
- `resnet`是一种残差网络
- 神经网络中`Momentum`指的是动量，动量系数通常被设定为0.9， 这个系数可以保证避免进入局部最优解，如果小球一直都往一个方向滚，那么动量将会积累，那么将不会落入到局部的最低的点而是直接冲出去。表达式是
```markdown
速度 = 动量系数 * 上一时刻的速度 + 学习率 * 当前速度
新参数 = 旧参数 - 速度
```
- 所以如果速度越快那么参数的更新就会越快，可以一定程度上避免局部最优解的出现。
- 必须要 ***先执行loss.backward()*** 语句，这样模型参数的梯度数据就会自动存储在参数的`.grad`属性之中，随后通过`optimizer.step()`读取梯度
- 关于是否将`requires_grad`设置为True,如果为True， 那么说明这一部分是需要进行梯度计算的，如果为False， 那么说明这些参数是`Frozen Parameters`，不需要计算这些参数的梯度
```python
x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad = True)

a = x + y
a.requires_grad == False

b = x + z
b.requires_grad == True
```
# 3. `Neural Networks`
- 前面提到的`autograd`可以用于定义模型与计算模型的微分，`nn.Module`里面会包括`layers, forward(input), 返回output`
- `torch.tensor.view(-1, 1)`的作用是将这个张量的大小改变为指定的大小，***-1在这里的意思是这个维度上的数据应该是自动计算而不用自己指定的*** 
- `optim`往往就是用来实现梯度下降的工具，可以选择SGD等等，这里使用的是`Adam`优化器。不同的是`Adam`使用了动量思想与自适应学习率，偏差校正等方法来优化了训练的结果。