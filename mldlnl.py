import tensorflow as tf
from numpy import array
import numpy as np
import warnings
print("Interacting with reality")
class LinReg(object):
    
    def __init__(self):
        pass
    
    def fit(self,x,y,lr=0.1,iter_no=80000,loss_fun="L2",lang="en"):
        self.__x=x
        self.__y=y
        self.__w=0.
        self.__b=0.
        self.__loss_fun=loss_fun
        self.__lr=lr
        self.__iter_no=iter_no
        
        self.__W=tf.Variable(self.__w)
        self.__B=tf.Variable(self.__b)
        self.__y_pred=self.__W*self.__x+self.__B
        

        if self.__loss_fun=="L2":
            self.__loss=tf.reduce_sum(tf.square(self.__y-self.__y_pred))
        elif self.__loss_fun=="L1":
            self.__loss=tf.reduce_sum(tf.abs(self.__y-self.__y_pred))
        self.__opt=tf.train.AdamOptimizer(self.__lr).minimize(self.__loss)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.__iter_no):
                sess.run(self.__opt)
                if i%500==0:
                    self.__l=sess.run(self.__loss)
                    print("Iteration",i,self.__loss_fun,"Loss="+str(self.__l))
            self.__w=sess.run(self.__W)
            self.__b=sess.run(self.__B)

    def get_variables(self):
        return self.__w,self.__b

    def calc(self,in_val):
        return self.__w*in_val+self.__b

    def save(self,file_n):
        with open(file_n,"w") as file:
            file.write(str(self.__w)+" "+str(self.__b))
            file.close()
            
    def restore(self,file_n):
        with open(file_n,"r") as file:
            var=file.read().split()
            self.__w=float(var[0])
            self.__b=float(var[1])

    def restore_manually(self,W,B):
        self.__w=W
        self.__b=B
    
class MultiLinReg(object):
    def __init__(self,n_of_params):
        self.__n_of_params=n_of_params
        self.____params=[]
        
        self.__params=[]
        for _ in range(self.__n_of_params+1):
           self.__params.append(tf.Variable(0.1))
           
    def fit(self,x,y,lr=0.05,iter_no=70000,loss_fun="L2"):
        self.__x=x
        self.__y=y
        self.__lr=lr
        self.__iter_no=iter_no
        self.__loss_fun=loss_fun
        
        self.__model=self.__params[-1]
        for i in range(self.__n_of_params):
            self.__model+=self.__params[i]*self.__x[i]
        
        if self.__loss_fun=="L2":
            self.__loss=tf.reduce_sum(tf.square(self.__y-self.__model))
        elif self.__loss_fun=="L1":
            self.__loss=tf.reduce_sum(tf.abs(self.__y-self.__model))
        
        self.__opt=tf.train.AdamOptimizer(self.__lr).minimize(self.__loss)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.__iter_no):
                sess.run(self.__opt)
                if i%500==0:
                    self.__l=sess.run(self.__loss)
                    print("Iteration",i,self.__loss_fun,"Loss="+str(self.__l))
            self.____params=[]
            for i in range(self.__n_of_params+1):
                self.____params.append(sess.run(self.__params[i]))

    def get_variables(self):
        return tuple(self.____params)

    def calc(self,in_val):
        val=self.____params[-1]
        for i in range(self.__n_of_params):
            val+=in_val[i]*self.____params[i]
        return val

    def save(self,file_n):
        with open(file_n,"w") as file:
            for i in range(self.__n_of_params+1):
                file.write(str(self.____params[i])+" ")
            file.close()

    def restore(self,file_n):
        self.____params=[]
        with open(file_n,"r") as file:
            params_r=file.read().split(" ")
            file.close()
            for i in range(self.__n_of_params):
                self.____params.append(float(params_r[i]))

class Perceptron(object):
    def __init__(self,neurons=[1,1,1],activation_fun="tanh"):
        self.__activation_fun=activation_fun.lower()
        self.__neurons=neurons
        self.__weights={}
        self.__biases={}
        self.__n_layers=len(self.__neurons)-1
        if self.__n_layers<2:
            raise AttributeError("Perceptron must have at least 3 layers")
        for i in range(self.__n_layers):
            self.__weights[i]=tf.Variable(tf.random.normal([neurons[i],neurons[i+1]]))
            self.__biases[i]=tf.Variable(tf.random.normal([neurons[i+1]]))

        self.__X=tf.compat.v1.placeholder(tf.float32,[None,self.__neurons[0]])
        self.__Y=tf.compat.v1.placeholder(tf.float32,[None,self.__neurons[-1]])
        self.__lr=tf.compat.v1.placeholder("float")
        self.__pkeep=tf.compat.v1.placeholder("float")
        
        if activation_fun.lower()=="tanh":
            self.__activator=tf.nn.tanh
        elif activation_fun.lower()=="relu":
            self.__activator=tf.nn.relu
        elif activation_fun.lower()=="sigmoid":
            self.__activator=tf.nn.sigmoid
        else:
            self.__activator=tf.nn.tanh
            raise AttributeError(activation_fun+" has not been defined yet; use tanh, relu or sigmoid instead")
        
        self.__layers=[]
        self.__layers.append(tf.nn.dropout(self.__activator(tf.linalg.matmul(self.__X,self.__weights[0])+self.__biases[0]),rate=1-self.__pkeep))
        for i in range(1,self.__n_layers-1):
            self.__layers.append(tf.nn.dropout(self.__activator(tf.linalg.matmul(self.__layers[i-1],self.__weights[i])+self.__biases[i]),rate=1-self.__pkeep))
            
        self.__layers.append(tf.nn.softmax(tf.linalg.matmul(self.__layers[self.__n_layers-2],self.__weights[self.__n_layers-1])+self.__biases[self.__n_layers-1])+1e-8)

        
        self.__xent=-tf.reduce_sum(self.__Y*tf.math.log(self.__layers[-1]))

        self.__correct_pred=tf.equal(tf.argmax(self.__Y,1),tf.argmax(self.__layers[-1],1))
        self.__accuracy=tf.reduce_mean(tf.cast(self.__correct_pred,tf.float32))

        self.__optimizer=tf.compat.v1.train.AdamOptimizer(self.__lr).minimize(self.__xent)

        self.__sess=tf.compat.v1.Session()
        self.__sess.run(tf.compat.v1.global_variables_initializer())

    def fit(self,x,y,epochs=5,batch_size=200,lr=0.01,keep_prob=1.):
        data_x=tools.split_batch(x,batch_size)
        data_y=tools.split_batch(y,batch_size)
        for j in range(epochs):
            for i in range(len(data_x)):
                self.__sess.run(self.__optimizer,feed_dict={self.__X:data_x[i],self.__Y:data_y[i],self.__lr:lr,self.__pkeep:keep_prob})
            acc,loss=self.__sess.run([self.__accuracy,self.__xent],feed_dict={self.__X:data_x[i],self.__Y:data_y[i],self.__lr:lr,self.__pkeep:1.})
            print("Epoch",j,"Cross Entropy Loss="+str(loss),"Accuracy="+str(acc))


    def calc(self,x,argmax=False):
        if len(x.shape)==1:
            if argmax==True:
                return self.__sess.run(tf.argmax(self.__layers[-1],1),feed_dict={self.__X:(x,),self.__pkeep:1.})
            else:
                return self.__sess.run(self.__layers[-1],feed_dict={self.__X:(x,),self.__pkeep:1.})
        else:
            if argmax==True:
                return self.__sess.run(tf.argmax(self.__layers[-1],1),feed_dict={self.__X:x,self.__pkeep:1.})
            else:
                return self.__sess.run(self.__layers[-1],feed_dict={self.__X:x,self.__pkeep:1.})
    def save(self,file):
        import os as __os
        import zipfile as __z
        for i in range(self.__n_layers):
            np.save("w"+str(i),self.__sess.run(self.__weights[i]))
            np.save("b"+str(i),self.__sess.run(self.__biases[i]))

        __meta=open("metadata","w")
        __meta.write(str(self.__n_layers)+"\n")
        __meta.write(self._activation_fun.lower())
        
        for k in range(len(self.__neurons)):
            __meta.write("\n"+str(self.__neurons[k]))
        
        __meta.close()
        
        zip_=__z.ZipFile(file,"w")
        for i in range(self.__n_layers):
            zip_.write("w"+str(i)+".npy")
            __os.remove("w"+str(i)+".npy")
            zip_.write("b"+str(i)+".npy")
            __os.remove("b"+str(i)+".npy")
                    
        zip_.write("metadata")
        __os.remove("metadata")
        
        zip_.close()  
        

    def restore(self,file):
        self.__sess.close()
        import os as __os
        import zipfile as __z
        z_=__z.ZipFile(file)
        z_.extractall()
        __meta=open("metadata","r").readlines()
        self.__n_layers=int(__meta[0][:-1])
        activation_fun=__meta[1][:-1]
        self.__activation_fun=activation_fun
        self.__neurons=[]

        for __i in __meta[2:]:
            self.__neurons.append(int(__i[:-1]))
        
        self.__neurons[-1]=int(__meta[-1])

        
        del __meta
        self.__weights,self.__biases={},{}
        self.__n_layers=len(self.__neurons)-1

        
        for i in range(self.__n_layers):
            self.__weights[i]=tf.Variable(np.load("w"+str(i)+".npy"))
            __os.remove("w"+str(i)+".npy")
            self.__biases[i]=tf.Variable(np.load("b"+str(i)+".npy"))
            __os.remove("b"+str(i)+".npy")
        self.__X=tf.compat.v1.placeholder(tf.float32,[None,self.__neurons[0]])
        self.__Y=tf.compat.v1.placeholder(tf.float32,[None,self.__neurons[-1]])
        self.__lr=tf.compat.v1.placeholder("float")
        self.__pkeep=tf.compat.v1.placeholder("float")
        
        if activation_fun=="tanh":
            self.__activator=tf.nn.tanh
        elif activation_fun=="relu":
            self.__activator=tf.nn.relu
        elif activation_fun=="sigmoid":
            self.__activator=tf.nn.sigmoid
        else:
            self.__activator=tf.nn.tanh
        
        self.__layers=[]
        self.__layers.append(tf.nn.dropout(self.__activator(tf.linalg.matmul(self.__X,self.__weights[0])+self.__biases[0]),rate=1-self.__pkeep))
        for i in range(1,self.__n_layers-1):
            self.__layers.append(tf.nn.dropout(self.__activator(tf.linalg.matmul(self.__layers[i-1],self.__weights[i])+self.__biases[i]),rate=1-self.__pkeep))
            
        self.__layers.append(tf.nn.softmax(tf.linalg.matmul(self.__layers[self.__n_layers-2],self.__weights[self.__n_layers-1])+self.__biases[self.__n_layers-1])+1e-8)

        
        self.__xent=-tf.reduce_sum(self.__Y*tf.math.log(self.__layers[-1]))

        self.__correct_pred=tf.equal(tf.argmax(self.__Y,1),tf.argmax(self.__layers[-1],1))
        self.__accuracy=tf.reduce_mean(tf.cast(self.__correct_pred,tf.float32))

        self.__optimizer=tf.compat.v1.train.AdamOptimizer(self.__lr).minimize(self.__xent)

        self.__sess=tf.compat.v1.Session()
        self.__sess.run(tf.compat.v1.global_variables_initializer())
    
    def test(self,x,y):
        acc,loss=self.__sess.run([self.__accuracy,self.__xent],feed_dict={self.__X:x,self.__Y:y,self.__pkeep:1.})
        return loss,acc
    
    def test(self,x,y):
        acc,loss=self.__sess.run([self.__accuracy,self.__xent],feed_dict={self.__X:x,self.__Y:y,self.__pkeep:1.})
        return loss,acc
    def properties(self):
        return (self.__neurons,self.__activation_fun)

class tools():
    @staticmethod
    def split_batch(x,batch_size):
        lenght=len(x)
        index_list=[]
        
        for i in range(0,lenght,batch_size):
            index_list.append(i)
        index_list.append(lenght)

        out=[]

        for i in range(len(index_list)-1):
            out.append(x[index_list[i]:index_list[i+1]])

        return np.array(out)


class ExpReg(object):
    def __init__(self):
        self.__X=tf.placeholder("float")
        self.__Y=tf.placeholder("float")
        self.__lr=tf.placeholder("float")

        self.__w=tf.Variable(0.)
        self.__model=tf.math.exp(tf.multiply(self.__X,self.__w))

        self.__error=tf.reduce_sum(tf.square(self.__Y-self.__model))
        self.__optimizer=tf.compat.v1.train.AdamOptimizer(self.__lr).minimize(self.__error)
        
        self.__sess=tf.Session()
        self.__sess.run(tf.global_variables_initializer())
    def fit(self,x,y,lr=0.01,iter_no=50000):
        for __i in range(iter_no):
            self.__sess.run(self.__optimizer,feed_dict={self.__X:x,self.__Y:y,self.__lr:lr})
            if __i%100==0:
                err=self.__sess.run(self.__error,feed_dict={self.__X:x,self.__Y:y})
                print("Iteration",__i,"L2 Loss="+str(err))

    def calc(self,x):
        return self.__sess.run(self.__model,feed_dict={self.__X:x})

    def get_variables(self):
        return self.__sess.run(self.__w)

    def save(self,n_of_file):
        __var=self.__sess.run(self.__w)
        with open(n_of_file,mode="w") as file:
            file.write(str(__var))
            file.close()

    def restore(self,n_of_file):
        with open(n_of_file) as file:
            __inner=float(file.read())
            file.close()
        
        self.__w=tf.Variable(__inner)
        self.__model=tf.math.exp(tf.multiply(self.__w   ,self.__X))
        self.__error=tf.reduce_sum(tf.square(self.__Y-self.__model))
        self.__optimizer=tf.compat.v1.train.AdamOptimizer(self.__lr).minimize(self.__error)
        self.__sess=tf.Session()
        self.__sess.run(tf.global_variables_initializer())       


class CExpReg(object):
    def __init__(self):
        self.__X=tf.placeholder("float")
        self.__Y=tf.placeholder("float")
        self.__lr=tf.placeholder("float")

        self.__w=tf.Variable(tf.truncated_normal([4],stddev=0.1))
        self.__model=tf.add(tf.multiply(self.__w[0],tf.math.exp(tf.add(tf.multiply(self.__w[2],self.__X),self.__w[3]))),self.__w[1])


        self.__error=tf.reduce_sum(tf.square(self.__Y-self.__model))
        self.__optimizer=tf.compat.v1.train.AdamOptimizer(self.__lr).minimize(self.__error)
        
        self.__sess=tf.Session()
        self.__sess.run(tf.global_variables_initializer())
    def fit(self,x,y,lr=0.01,iter_no=50000):
        for __i in range(iter_no):
            self.__sess.run(self.__optimizer,feed_dict={self.__X:x,self.__Y:y,self.__lr:lr})
            if __i%100==0:
                err=self.__sess.run(self.__error,feed_dict={self.__X:x,self.__Y:y})
                print("Iteration",__i,"L2 Loss="+str(err))

    def calc(self,x):
        return self.__sess.run(self.__model,feed_dict={self.__X:x})

    def get_variables(self):
        return self.__sess.run(self.__w)

    def save(self,n_of_file):
        __var=self.__sess.run(self.__w)
        with open(n_of_file,mode="w") as file:
            __out=""
            for __i in range(0,4):
                __out+=str(__var[__i])+" "
            file.write(__out[:-1])
            file.close()

    def restore(self,n_of_file):
        with open(n_of_file) as file:
            __inner=file.read().split(" ")
            __inner=[float(__var) for __var in __inner]
            file.close()
        
        self.__w=tf.Variable(__inner)
        self.__model=tf.add(tf.multiply(self.__w[0],tf.math.exp(tf.add(tf.multiply(self.__w[2],self.__X),self.__w[3]))),self.__w[1])
        self.__error=tf.reduce_sum(tf.square(self.__Y-self.__model))
        self.__optimizer=tf.compat.v1.train.AdamOptimizer(self.__lr).minimize(self.__error)
        self.__sess=tf.Session()
        self.__sess.run(tf.global_variables_initializer())

print("Using Tensorflow backend")
