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
         |---MultiLinReg(n_of_params)--|
                                  |--fit(x,y,lr=0.05,iter_no=70000,loss_fun="L2",lang="en")
                                  |--get_variables()
                                  |--calc(x)
                                  |--save(file_name)
                                  |--restore(file_name)                                   
</pre>

<ol>
     <li><pre><b>LinReg:</b></pre></li>
           <pre><b>type:object</b>Linear Regression object. Use this object to create linear regression models.</pre> 
     <ul><pre><b>fit(x,y,lr=0.1,iter_no=80000,loss_fun="L2",lang="en")</b></pre></li><p><b>type:method</b> fit() optimizes model with specified loss function. It uses <code>tf.train.AdamOptimizer to find optimum weight and bias</code>.<code>x</code> is input,<code>y</code> is output. <code>lr</code> islearning rate, it's default 0.1.<code>iter_no</code> is number of train step.<code>loss_fun</code> is a string represents loss function.It's default L2, you can also use L1 with <code>"L1"</code>. <code>lang</code> is language that used for getting metrics. Just English and Turkish supported. It will be removed in future version.</p>
     <ul><pre><b>get_variables() </b></pre></li><p><b>type:method</b> The function exports variables and returns a tuple<code>weight,bias)</code></p>
     <ul><pre><b>calc(x) </b></pre></li><pre><b>type:method</b></pre><p> Computes the output valur with spesific input. </p>
     <ul><pre><b>save(file_n) </b></pre></li><pre><b>type:method</b></pre><p>Exports and saves parametrs into a file.<code>file_n</code> is a string takes name of file. Don't add directory into the string.</p>
     <ul><pre><b>restore(file_n) </b></pre></li><pre><b>type:method</b></pre><p>Imports and restores parameters from file.<code>file_n</code> is a string takes name of file. Don't add directory into the string. </p>
     <ul><pre><b>restore_manually(W,B) </b></pre></li><pre><b>type:method</b></pre><p> This function utilizes restore parameters manually. You can initialize variables by this function.</p>
     <li><pre><b>MultiLinReg(n_of_params):</b> Multi Linear Regression object. Use this object to create multi linear regression models.</pre></li> 
     <pre><b>type:object</b></pre><p> <code>n_of_params</code> is number of parameters.</p>
<ul><pre><b>fit(x,y,lr=0.1,iter_no=80000,loss_fun="L2",lang="en")</b></pre></li><p><b>type:method</b> fit() optimizes model with specified loss function. It uses <code>tf.train.AdamOptimizer to find optimum weight and bias</code>.<code>x</code> is input,<code>y</code> is output. <code>lr</code> islearning rate, it's default 0.1.<code>iter_no</code> is number of train step.<code>loss_fun</code> is a string represents loss function.It's default L2, you can also use L1 with <code>"L1"</code>. <code>lang</code> is language that used for getting metrics. Just English and Turkish supported. It will be removed in future version.</p>
     <ul><pre><b>get_variables() </b></pre></li><p><b>type:method</b> The function exports variables and returns a tuple<code>weight,bias)</code></p>
     <ul><pre><b>calc(x) </b></pre></li><pre><b>type:method</b></pre><p> Computes the output valur with spesific input. </p>
     <ul><pre><b>save(file_n) </b></pre></li><pre><b>type:method</b></pre><p>Exports and saves parametrs into a file.<code>file_n</code> is a string takes name of file. Don't add directory into the string.</p>
     <ul><pre><b>restore(file_n) </b></pre></li><pre><b>type:method</b></pre><p>Imports and restores parameters from file.<code>file_n</code> is a string takes name of file. Don't add directory into the string. </p>
</ol>

