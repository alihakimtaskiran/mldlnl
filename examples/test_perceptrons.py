import mldlnl
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("data/mnist",one_hot=True)

nn=mldlnl.Perceptron([784,256,10],"relu")
nn.fit(mnist.train.images,mnist.train.labels,epochs=10,lr=0.001,keep_prob=0.5)
loss,acc=nn.test(mnist.test.images,mnist.test.labels)
print("Test loss="+str(loss)," accuracy="+str(acc))

print(nn.calc(mnist.test.images[7:10],argmax=True))
nn.save("perceptron")

p=mldlnl.Perceptron([784,256,10],"relu")
print(p.calc(mnist.test.images[7:10],argmax=True))
p.restore("perceptron")
print(p.calc(mnist.test.images[7:10],argmax=True))

