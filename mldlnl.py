import tensorflow as tf
from random import random
from numpy import array
print("Using Tensorflow backend")
class LinReg(object):
    
    def __init__(self):
        self.metrics=[]
    
    def fit(self,x,y,lr=0.1,iter_no=80000,loss_fun="L2",lang="en"):
        self.x=x
        self.y=y
        self.w=random()
        self.b=random()
        self.loss_fun=loss_fun
        self.lr=lr
        self.iter_no=iter_no
        self.lang=lang
        #y=W*x+B Define the parameters
        self.W=tf.Variable(self.w)
        self.B=tf.Variable(self.b)
        self.y_pred=self.W*self.x+self.B
        #select loss function

        if self.loss_fun=="L2":
            self.loss=tf.reduce_sum(tf.square(self.y-self.y_pred))
        elif self.loss_fun=="L1":
            self.loss=tf.reduce_sum(tf.abs(self.y-self.y_pred))
        self.opt=tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        #tf.Session process
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.iter_no):
                sess.run(self.opt)
                if i%500==0:
                    if self.lang=="en":
                        self.l=sess.run(self.loss)
                        print("Iteration",i,self.loss_fun,"Loss="+str(self.l))
                    elif self.lang=="tr":
                        self.l=sess.run(self.loss)
                        print("Döngü",i,self.loss_fun,"Loss="+str(self.l))
            self.w=sess.run(self.W)
            self.b=sess.run(self.B)

    def get_variables(self):
        return self.w,self.b

    def calc(self,in_val):
        return self.w*in_val+self.b

    def save(self,file_n):
        with open(file_n,"w") as file:
            file.write(str(self.w)+" "+str(self.b))
            file.close()
            
    def restore(self,file_n):
        with open(file_n,"r") as file:
            var=file.read().split()
            self.w=float(var[0])
            self.b=float(var[1])

    def restore_manually(self,W,B):
        self.w=W
        self.b=B
    
class MultiLinReg(object):
    def __init__(self,n_of_params):
        self.n_of_params=n_of_params
        self.__params=[]
        #create parameters
        self.params=[]
        self.metrics=[]
        for _ in range(self.n_of_params+1):
           self.params.append(tf.Variable(random()))
           
    def fit(self,x,y,lr=0.05,iter_no=70000,loss_fun="L2",lang="en"):
        self.x=x
        self.y=y
        self.lr=lr
        self.iter_no=iter_no
        self.loss_fun=loss_fun
        self.lang=lang
        
        self.model=self.params[-1]#add the last parameter which called bias
        for i in range(self.n_of_params):
            self.model+=self.params[i]*self.x[i]
        #choose loss function
        if self.loss_fun=="L2":
            self.loss=tf.reduce_sum(tf.square(self.y-self.model))
        elif self.loss_fun=="L1":
            self.loss=tf.reduce_sum(tf.abs(self.y-self.model))
        #define optimizer
        self.opt=tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        #tf session computations
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.iter_no):
                sess.run(self.opt)
                if i%500==0:
                    self.l=sess.run(self.loss)
                    self.metrics.append(l)
                    if lang=="en":
                        print("Iteration",i,self.loss_fun,"Loss="+str(self.l))
                    elif lang=="tr":
                        print("Döngü",i,self.loss_fun,"Loss="+str(self.l))
            self.__params=[]
            for i in range(self.n_of_params+1):
                self.__params.append(sess.run(self.params[i]))

    def get_variables(self):
        return tuple(self.__params)

    def calc(self,in_val):
        val=self.__params[-1]
        for i in range(self.n_of_params):
            val+=in_val[i]*self.__params[i]
        return val

    def save(self,file_n):
        with open(file_n,"w") as file:
            for i in range(self.n_of_params+1):
                file.write(str(self.__params[i])+" ")
            file.close()

    def restore(self,file_n):
        self.__params=[]
        with open(file_n,"r") as file:
            params_r=file.read().split(" ")#readed parameters
            file.close()
            for i in range(self.n_of_params):
                self.__params.append(float(params_r[i]))

