import numpy as np
import matplotlib.pyplot as plt

# 1. 定义函数f(x) = sin(x)
def true_function(x):
    return np.sin(x)

# 2. 生成训练集和测试集
# 训练集
x_train = np.linspace(-5, 5, 100)
y_train = true_function(x_train)

# 测试集
x_test = np.linspace(-5, 5, 100)
y_test = true_function(x_test)

# 3. 定义一个两层的 ReLU 神经网络
class Relu:
    def __init__(self):
        self.mem = {}
        
    def forward(self, x):
        self.mem['x'] = x
        return np.where(x > 0, x, np.zeros_like(x))
    
    def backward(self, grad_y):
        x = self.mem['x']
        grad_x = grad_y * (x > 0)  # ReLU 梯度：当 x > 0 时为 grad_y，否则为 0
        return grad_x

class ReLUNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01  # 添加较小的初始化权重
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01  # 添加较小的初始化权重
        self.b2 = np.zeros(output_size)
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        # 第一层前向传播
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        # 第二层前向传播
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2

    def backward(self, x, y, learning_rate=0.01):
        # 计算损失函数的梯度
        m = y.shape[0]
        output = self.forward(x)
        
        # 损失函数的导数（均方误差）
        d_loss = 2 * (output - y) / m
        
        # 计算第二层的梯度
        d_W2 = np.dot(self.a1.T, d_loss)
        d_b2 = np.sum(d_loss, axis=0)
        
        # 计算第一层的梯度
        d_a1 = np.dot(d_loss, self.W2.T)
        d_z1 = d_a1 * (self.z1 > 0)  # ReLU 导数
        
        d_W1 = np.dot(x.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0)
        
        # 更新参数
        self.W1 -= learning_rate * d_W1
        self.b1 -= learning_rate * d_b1
        self.W2 -= learning_rate * d_W2
        self.b2 -= learning_rate * d_b2

# 4. 训练神经网络
model = ReLUNetwork(input_size=1, hidden_size=50, output_size=1)

# 转换为列向量
x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

# 训练模型
epochs = 10000  # 可以根据需要调整训练周期
for epoch in range(epochs):
    model.backward(x_train, y_train)
    if epoch % 1000 == 0:
        loss = np.mean((model.forward(x_train) - y_train) ** 2)
        print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")


# 5. 测试拟合效果
x_test = x_test.reshape(-1, 1)
y_test_pred = model.forward(x_test)

# 绘制拟合结果
plt.figure(figsize=(10, 6))
plt.plot(x_test, y_test, label="True Function (sin(x))", color='blue')
plt.plot(x_test, model.forward(x_test), label="Fitted Function", color='red', linestyle='--')
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Function Fitting using ReLU Neural Network")
plt.show()