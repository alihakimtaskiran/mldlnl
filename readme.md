<h1>Documentation</h1>
<h2>1. What's the MLDLNL?</h2>
<pre>  MLDLNL is a tensorflow based high level API. It facilitates creating machine learning models(for now).</pre>

<h2>2.Installation</h2>
<h3>a.Requirements</h3>
<pre>-Python
     -Numpy
     -Tensorflow 1.x.x</pre>
<h3>b.Installation Steps</h3>

<font size="7"><b><p>  You can run the module including into working directory. 
You can also install it into python3. After the installation, you can run the module without including into working directory.</p></b></font>
Follow this steps for install the module(optional)
<h3>·Installation Steps for Linux</h3>
<ol>
  <li>Open the terminal. After that</li>
  <li><code>git clone https://github.com/alihakimtaskiran/mldlnl.git</code></li>
  <li><code>cd "mldlnl"</code></li>
  <li><code>sudo cp vsm.py /usr/lib/python3.6/</code></li>
  <li>If you haven't installed numpy, install the numpy with <code>pip3 install numpy</code></li>
  <li>If you haven't installed tensorflow 1.x, install the tensorflow 1.x with <code>pip3 install tensorflow==1.15.0</code></li>
 </ol>
<p>Finally, you can use the module in python3 just one lines of code:<code>import mldlnl</code></p>

<h2>3.Tree of Module</h2>
<pre>
mldlnl---|
         |
         |---LinReg()--|
         |             |--fit(x,y,lr=0.1,iter_no=80000,loss_fun="L2",lang="en")
         |             |--get_variables()
         |             |--calc(x)
         |             |--save(file_name)
         |             |--restore(file_name)
         |             |--restore_manually(weight,bias)
         |
         |
         |---LogReg(n_of_params)--|
                                  |--fit(x,y,lr=0.05,iter_no=70000,loss_fun="L2",lang="en")
                                  |--get_variables()
                                  |--calc(x)
                                  |--save(file_name)
                                  |--restore(file_name)                                   
</pre>
