import mldlnl
import numpy as np
x=np.array((np.arange(10,100),np.arange(0,90),np.arange(-80,10),np.arange(110,200),np.arange(90,180)))
y=67*x[0]+78*x[1]+597*x[2]+6*x[3]-8*x[4]-98

Line=mldlnl.MultiLinReg(5)
Line.fit(x,y,lr=0.01,iter_no=10000)
Line.save("var.txt")
print(Line.get_variables())
print(Line.calc(np.zeros(5)))
del Line

Line_1=mldlnl.MultiLinReg(5)
Line_1.restore("var.txt")
print(Line_1.get_variables())
print(Line_1.calc(np.zeros(5)))
Line_1.fit(x,y,lr=0.01,iter_no=100001)
Line_1.save("var.txt")

