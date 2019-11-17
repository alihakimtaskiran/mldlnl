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
  <li><code>sudo cp mldlnl.py /usr/lib/python3.6/</code></li>
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
         |---MultiLinReg(n_of_params)--|
                                  |--fit(x,y,lr=0.05,iter_no=70000,loss_fun="L2",lang="en")
                                  |--get_variables()
                                  |--calc(x)
                                  |--save(file_name)
                                  |--restore(file_name)                                   
</pre>
<hr>
<ul>
     <li>
     <pre><b>LinReg:</b></pre></li>
           <pre><b>type:object</b>Linear Regression object. Use this object to create linear regression models.</pre> 
     <ul><pre><b>fit(x,y,lr=0.1,iter_no=80000,loss_fun="L2",lang="en")</b></pre></li><p><b>type:method</b> fit() optimizes model with specified loss function. It uses <code>tf.train.AdamOptimizer to find optimum weight and bias</code>.<code>x</code> is input,<code>y</code> is output. <code>lr</code> islearning rate, it's default 0.1.<code>iter_no</code> is number of train step.<code>loss_fun</code> is a string represents loss function.It's default L2, you can also use L1 with <code>"L1"</code>.</p></ul>
     <ul><pre><b>get_variables() </b></pre></li><p><b>type:method</b> The function exports variables and returns a tuple<code>weight,bias)</code></p></ul>
     <ul><pre><b>calc(x) </b></pre></li><pre><b>type:method</b></pre><p> Computes the output valur with spesific input. </p></ul>
     <ul><pre><b>save(file_n) </b></pre></li><pre><b>type:method</b></pre><p>Exports and saves parametrs into a file.<code>file_n</code> is a string takes name of file. Don't add directory into the string.</p></ul>
     <ul><pre><b>restore(file_n) </b></pre></li><pre><b>type:method</b></pre><p>Imports and restores parameters from file.<code>file_n</code> is a string takes name of file. Don't add directory into the string. </p></ul>
     <ul><pre><b>restore_manually(W,B) </b></pre></li><pre><b>type:method</b></pre><p> This function utilizes restore parameters manually. You can initialize variables by this function.</p></ul>
     </li>
     <br>
     <hr>
     <br>
     <li>
     <pre><b>MultiLinReg(n_of_params):</b> Multi Linear Regression object. Use this object to create multi linear regression models.</pre></li> 
     <pre><b>type:object</b></pre><p> <code>n_of_params</code> is number of parameters.</p>
<ul>
     <pre><b>fit(x,y,lr=0.1,iter_no=80000,loss_fun="L2",lang="en")</b></pre></li><p><b>type:method</b> fit() optimizes model with specified loss function. It uses <code>tf.train.AdamOptimizer to find optimum weight and bias</code>.<code>x</code> is input,<code>y</code> is output. <code>lr</code> islearning rate, it's default 0.1.<code>iter_no</code> is number of train step.<code>loss_fun</code> is a string represents loss function.It's default L2, you can also use L1 with <code>"L1"</code>.</p></ul>
     <ul><pre><b>get_variables() </b></pre></li><p><b>type:method</b> The function exports variables and returns a tuple<code>weight,bias)</code></p></ul>
     <ul><pre><b>calc(x) </b></pre></li><pre><b>type:method</b></pre><p> Computes the output valur with spesific input. </p></ul>
     <ul><pre><b>save(file_n) </b></pre></li><pre><b>type:method</b></pre><p>Exports and saves parametrs into a file.<code>file_n</code> is a string takes name of file. Don't add directory into the string.</p></ul>
     <ul><pre><b>restore(file_n) </b></pre></li><pre><b>type:method</b></pre><p>Imports and restores parameters from file.<code>file_n</code> is a string takes name of file. Don't add directory into the string. </p></ul>
</li>
 <br>
     <hr>
     <br>
     <li>
     <pre><b>Perceptron(neurons,activation_fun="tanh")</b> Perceptron object utilizes creating multi layer perceptrons with specified activation function </pre></li> 
     <pre><b>type:object</b></pre><p> <code>neurons</code> is a list(like <code>[784,256,128,10]</code>) to represent number of neuron per layer. The list should have at least 3 elements.<code>activation_fun</code> is specified activation function for perceptron.<code>tanh</code>,<code>ReLU</code> and <code>sigmoid</code> are supported activation functions. Activation function of last layer is <code>softmax</code> independed from specified activation function.</p>
<ul>
     <ul><pre><b>fit(x,y,epochs=5,batch_size=200,lr=0.01,keep_prob=1.) </b></pre></li><p><b>type:method </b> <code><b>fit()</b></code> utilizes train perceptron. It uses AdamOptimizer to optimize model.<code><b>x</b></code> is input data and <code><b>y</b></code> is output data to train perceptron. You don't need to split into batchs the data.<code> Percepton <code> object has internal batch splitting system.<code><b>epochs</b></code> is training epochs.<code><b>batch_size</b></code> is default set into 200.<code><b>lr</b></code> is default set into 0.01.<code><b>keep_prob</b> is probibilty of retained neurons after dropout </code> </p></ul>
     <ul><pre><b>calc(x,argmax=False) </b></pre></li><p><b>type: method</b> This function feed forwards an input value. You can compute the output of perceptron.<code><b>x</b></code> is input. <code><b>argmax</b></code> is a boolean. If it's True, the function returns index of maximum value of percpetron's output. If it's False, function returns output of perceptron.</p></ul>
     <ul><pre><b>save(file) </b></pre></li><p><b>type:method </b>It exports parameters of perceptrons into a file<code><b>file</b></code> is name of the file.</p></ul>
     <ul><pre><b>restore(file) </b></pre></li><p><b>type:method </b>It imports parameters of perceptron from file. <code><b>file</b></code> is name of the file.</p></ul>
     
</li>

<br>
<br>
     <hr>
     <br>
     <li>
     <pre><b>tools()</b> This class contains usefull tools for data science.</pre></li> 
     <pre><b>type:class</b></pre>
     <ul><pre><b>split_batch(x,batch_size) </b></pre></li><p><b>type:method</b> This function facilates spliting dataset into batchs.<code><b>x</b></code> is input data, <code><b>batch_size</b></code> is size of batch, it's an integer.</p></ul>
<ul>
      </li>
