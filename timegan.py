"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

timegan.py

Note: Use original data as training set to generater synthetic data (time-series)
"""
import time

# Necessary Packages
import tensorflow as tf

from TimeGAN.data_loading import real_data_loading2


import numpy as np
from utils import extract_time, rnn_cell, random_generator, batch_generator
import os


def timegan (ori_data, parameters,model_path,train=True,result_path=None):
  """TimeGAN function.
  
  Use original data as training set to generater synthetic data (time-series)
  
  Args:
    - ori_data: original time-series data
    - parameters: TimeGAN network parameters
    
  Returns:
    - generated_data: generated time-series data
  """
  result_path=result_path+'/'+'time.txt'

  # Initialization on the Graph
  tf.reset_default_graph()

  # Basic Parameters
  no, seq_len, dim = np.asarray(ori_data).shape
    
  # Maximum sequence length and each sequence length
  ori_time, max_seq_len = extract_time(ori_data)
  
  def MinMaxScaler(data):
    """Min-Max Normalizer.
    
    Args:
      - data: raw data
      
    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """    
    min_val = np.min(np.min(data, axis = 0), axis = 0)
    data = data - min_val
      
    max_val = np.max(np.max(data, axis = 0), axis = 0)
    norm_data = data / (max_val + 1e-7)
      
    return norm_data, min_val, max_val
  
  # Normalization
  ori_data, min_val, max_val = MinMaxScaler(ori_data)
              
  ## Build a RNN networks          
  
  # Network Parameters
  hidden_dim   = parameters['hidden_dim'] 
  num_layers   = parameters['num_layer']
  iterations   = parameters['iterations']
  batch_size   = parameters['batch_size']
  module_name  = parameters['module'] 
  z_dim        = dim
  gamma        = 1
    
  # Input place holders
  X = tf.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x")
  Z = tf.placeholder(tf.float32, [None, max_seq_len, z_dim], name = "myinput_z")
  T = tf.placeholder(tf.int32, [None], name = "myinput_t")

  def dilated_causal_conv1d(X, hidden_dim, kernel_size, dilation_rate, scope):
    """扩张因果卷积（Dilated Causal Convolution）"""
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
      # 计算填充大小，确保因果性
      pad_size = (kernel_size - 1) * dilation_rate
      X_padded = tf.pad(X, [[0, 0], [pad_size, 0], [0, 0]])  # 只在左侧填充
      # 创建卷积核 (kernel_size, input_dim, filters)
      input_dim = X.shape[-1]
      W = tf.compat.v1.get_variable(
        name=f"W_dilated_{dilation_rate}",
        shape=[kernel_size, input_dim, hidden_dim],
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)
      )
      b = tf.compat.v1.get_variable(
        name=f"b_dilated_{dilation_rate}",
        shape=[hidden_dim],
        initializer=tf.compat.v1.zeros_initializer()
      )
      # 进行 1D 卷积
      conv_out = tf.nn.conv1d(X_padded, W, stride=1, padding="VALID", dilations=dilation_rate) + b
    return tf.nn.relu(conv_out)  # ReLU 激活

  def embedder(X, T):
    """Embedding network between original feature space to latent space using DCCN.

    Args:
      - X: input time-series features
      - T: input time information

    Returns:
      - H: embeddings
    """
    dilation_rates = [1, 2,4, 8]
    kernel_size = 3
    #1
    # with tf.compat.v1.variable_scope("embedder", reuse=tf.compat.v1.AUTO_REUSE):
    #   dilation_rates = [1, 2, 4, 8]
    #   conv_outputs = X
    #   for rate in dilation_rates:
    #     conv_outputs = tf.compat.v1.layers.conv1d(
    #       conv_outputs,
    #       filters=hidden_dim,
    #       kernel_size=3,
    #       dilation_rate=rate,
    #       padding="causal",
    #       activation=tf.nn.relu
    #     )
    #   H = tf.compat.v1.layers.dense(conv_outputs, hidden_dim, activation=tf.nn.sigmoid)
    #2
    with tf.variable_scope("embedder", reuse = tf.AUTO_REUSE):
      e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length = T)
      H = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
    #3
    # with tf.compat.v1.variable_scope("embedder", reuse=tf.compat.v1.AUTO_REUSE):
    #   conv_outputs = X  # 初始输入
    #   for rate in dilation_rates:
    #     conv_outputs = dilated_causal_conv1d(conv_outputs, hidden_dim, kernel_size, rate, scope="conv_layer")
    #
    #   # 最终投影到潜在空间
    #   W_proj = tf.compat.v1.get_variable(
    #     name="W_proj",
    #     shape=[hidden_dim, hidden_dim],
    #     initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)
    #   )
    #   b_proj = tf.compat.v1.get_variable(
    #     name="b_proj",
    #     shape=[hidden_dim],
    #     initializer=tf.compat.v1.zeros_initializer()
    #   )
    #   H = tf.nn.sigmoid(tf.tensordot(conv_outputs, W_proj, axes=[-1, 0]) + b_proj)  # 矩阵运算代替 matmul
    return H


  def recovery(H, T):
    """Recovery network from latent space to original space using attention-based decoder.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - X_tilde: recovered data
    """
    # with tf.compat.v1.variable_scope("recovery", reuse=tf.compat.v1.AUTO_REUSE):
    #   attention_scores = tf.nn.softmax(tf.matmul(H, tf.transpose(H, perm=[0, 2, 1])))
    #   attention_outputs = tf.matmul(attention_scores, H)
    #   X_tilde = tf.compat.v1.layers.dense(attention_outputs, dim, activation=tf.nn.sigmoid)
    with tf.variable_scope("recovery", reuse = tf.AUTO_REUSE):
        r_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
        r_outputs, r_last_states = tf.nn.dynamic_rnn(r_cell, H, dtype=tf.float32, sequence_length = T)
        X_tilde = tf.contrib.layers.fully_connected(r_outputs, dim, activation_fn=tf.nn.sigmoid)

    # with tf.compat.v1.variable_scope("recovery", reuse=tf.compat.v1.AUTO_REUSE):
    #   # 1. 双向 RNN 作为编码器
    #   encoder_cells = [tf.compat.v1.nn.rnn_cell.GRUCell(hidden_dim) for _ in range(num_layers)]
    #   encoder_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(encoder_cells)
    #   encoder_outputs, encoder_final_state = tf.compat.v1.nn.dynamic_rnn(
    #     encoder_cell, H, dtype=tf.float32, sequence_length=T
    #   )
    #   # 2. 注意力层计算权重
    #   attention_layer = tf.keras.layers.Attention()
    #   context_vector = attention_layer([encoder_outputs, encoder_outputs])  # 自注意力
    #   # 3. 解码器：使用带注意力的 RNN
    #   decoder_cells = [tf.compat.v1.nn.rnn_cell.GRUCell(hidden_dim) for _ in range(num_layers)]
    #   decoder_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(decoder_cells)
    #   decoder_outputs, _ = tf.compat.v1.nn.dynamic_rnn(
    #     decoder_cell, context_vector, dtype=tf.float32, sequence_length=T
    #   )
    #   # 4. 全连接层投影回原始数据维度
    #   X_tilde = tf.compat.v1.layers.dense(decoder_outputs, dim, activation=tf.nn.sigmoid)
    return X_tilde
    


  def generator(Z, T):
    """Generator function using BiRNN.

    Args:
      - Z: random variables
      - T: input time information

    Returns:
      - E: generated embedding
    """
    # with tf.compat.v1.variable_scope("generator", reuse=tf.compat.v1.AUTO_REUSE):
    #   forward_cell = tf.compat.v1.nn.rnn_cell.GRUCell(hidden_dim)
    #   backward_cell = tf.compat.v1.nn.rnn_cell.GRUCell(hidden_dim)
    #   outputs, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, Z, dtype=tf.float32,
    #                                                          sequence_length=T)
    #   outputs_concat = tf.concat(outputs, axis=-1)
    #   E = tf.compat.v1.layers.dense(outputs_concat, hidden_dim, activation=tf.nn.sigmoid)
    with tf.variable_scope("generator", reuse = tf.AUTO_REUSE):
        e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
        e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, Z, dtype=tf.float32, sequence_length = T)
        E = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
    return E

  def supervisor (H, T):
    """Generate next sequence using the previous sequence.
    
    Args:
      - H: latent representation
      - T: input time information
      
    Returns:
      - S: generated sequence based on the latent representations generated by the generator
    """          
    with tf.variable_scope("supervisor", reuse = tf.AUTO_REUSE):
      e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers-1)])
      e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, H, dtype=tf.float32, sequence_length = T)
      S = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
    return S
          
  def discriminator (H, T):
    """Discriminate the original and synthetic time-series data.
    
    Args:
      - H: latent representation
      - T: input time information
      
    Returns:
      - Y_hat: classification results between original and synthetic time-series
    """        
    with tf.variable_scope("discriminator", reuse = tf.AUTO_REUSE):
      d_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, H, dtype=tf.float32, sequence_length = T)
      Y_hat = tf.contrib.layers.fully_connected(d_outputs, 1, activation_fn=None)
    return Y_hat   
    
  # Embedder & Recovery
  H = embedder(X, T)
  X_tilde = recovery(H, T)
    
  # Generator
  E_hat = generator(Z, T)
  H_hat = supervisor(E_hat, T)
  H_hat_supervise = supervisor(H, T)
    
  # Synthetic data
  X_hat = recovery(H_hat, T)
    
  # Discriminator
  Y_fake = discriminator(H_hat, T)
  Y_real = discriminator(H, T)     
  Y_fake_e = discriminator(E_hat, T)
    
  # Variables        
  e_vars = [v for v in tf.trainable_variables() if v.name.startswith('embedder')]
  r_vars = [v for v in tf.trainable_variables() if v.name.startswith('recovery')]
  g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
  s_vars = [v for v in tf.trainable_variables() if v.name.startswith('supervisor')]
  d_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
    
  # Discriminator loss
  D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
  D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
  D_loss_fake_e = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
  D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
            
  # Generator loss
  # 1. Adversarial loss
  G_loss_U = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
  G_loss_U_e = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)
    
  # 2. Supervised loss
  G_loss_S = tf.losses.mean_squared_error(H[:,1:,:], H_hat_supervise[:,:-1,:])
    
  # 3. Two Momments
  G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
  G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X,[0])[0])))
    
  G_loss_V = G_loss_V1 + G_loss_V2
    
  # 4. Summation
  G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100*G_loss_V 
            
  # Embedder network loss
  E_loss_T0 = tf.losses.mean_squared_error(X, X_tilde)
  E_loss0 = 10*tf.sqrt(E_loss_T0)
  E_loss = E_loss0  + 0.1*G_loss_S
    
  # optimizer
  E0_solver = tf.train.AdamOptimizer().minimize(E_loss0, var_list = e_vars + r_vars)
  E_solver = tf.train.AdamOptimizer().minimize(E_loss, var_list = e_vars + r_vars)
  D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list = d_vars)
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list = g_vars + s_vars)
  GS_solver = tf.train.AdamOptimizer().minimize(G_loss_S, var_list = g_vars + s_vars)

  saver = tf.train.Saver()
  ## TimeGAN training
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  #
  # if load_model  and os.path.exists(model_path + ".meta"):
  #   print(f"Loading model from {model_path}...")
  #   saver.restore(sess, model_path)
  #   print(f"Model restored from {model_path}")

  if train :
    start_train=time.time()
    print('未检测到模型或者继续训练模型')
    # 1. Embedding network training
    print('Start Embedding Network Training')
    for itt in range(iterations):
      # Set mini-batch
      X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
      # Train embedder
      _, step_e_loss = sess.run([E0_solver, E_loss_T0], feed_dict={X: X_mb, T: T_mb})
      # Checkpoint
      if itt % 1000 == 0:
        print('step: ' + str(itt) + '/' + str(iterations) + ', e_loss: ' + str(np.round(np.sqrt(step_e_loss), 4)))

    print('Finish Embedding Network Training')

    # 2. Training only with supervised loss
    print('Start Training with Supervised Loss Only')

    for itt in range(iterations):
      # Set mini-batch
      X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
      # Random vector generation
      Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
      # Train generator
      _, step_g_loss_s = sess.run([GS_solver, G_loss_S], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
      # Checkpoint
      if itt % 1000 == 0:
        print('step: ' + str(itt) + '/' + str(iterations) + ', s_loss: ' + str(np.round(np.sqrt(step_g_loss_s), 4)))

    print('Finish Training with Supervised Loss Only')

    # 3. Joint Training
    print('Start Joint Training')

    for itt in range(iterations):
      # Generator training (twice more than discriminator training)
      for kk in range(2):
        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        # Random vector generation
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        # Train generator
        _, step_g_loss_u, step_g_loss_s, step_g_loss_v = sess.run([G_solver, G_loss_U, G_loss_S, G_loss_V],
                                                                  feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
        # Train embedder
        _, step_e_loss_t0 = sess.run([E_solver, E_loss_T0], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})

      # Discriminator training
      # Set mini-batch
      X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
      # Random vector generation
      Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
      # Check discriminator loss before updating
      check_d_loss = sess.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
      # Train discriminator (only when the discriminator does not work well)
      if (check_d_loss > 0.15):
        _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})

      # Print multiple checkpoints
      if itt % 1000 == 0:
        print('step: ' + str(itt) + '/' + str(iterations) +
              ', d_loss: ' + str(np.round(step_d_loss, 4)) +
              ', g_loss_u: ' + str(np.round(step_g_loss_u, 4)) +
              ', g_loss_s: ' + str(np.round(np.sqrt(step_g_loss_s), 4)) +
              ', g_loss_v: ' + str(np.round(step_g_loss_v, 4)) +
              ', e_loss_t0: ' + str(np.round(np.sqrt(step_e_loss_t0), 4)))
    print('Finish Joint Training')
    if model_path is not None:
      print(f"Saving model to {model_path}...")
      saver.save(sess, model_path)
      print(f"Model saved at {model_path}")
    end_train=time.time()
    train_time=end_train-start_train
    print(f">>>>>>>>>>>>>>>>>>>>>>模型已保存,用时:{train_time / 60:.4f} min<<<<<<<<<<<<<<<<<<")
    with open(result_path, 'a') as file:  # 使用 'w' 模式，每次都会覆盖文件
      file.write(f"模型已保存,用时:{train_time / 60:.4f} min")  # 每个结果之间换行
  else:
    print(f"Loading model from {model_path}...")
    saver.restore(sess, model_path)
    print(f"Model restored from {model_path}")
    
  ## Synthetic data generation
  Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
  start_gen=time.time()
  # ori_data=real_data_loading2('train', 24,'Sales','S')
  # for i in range(no):
  #   ori_time.append(24)
  generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data, T: ori_time})
  print(len(generated_data_curr))
    
  generated_data = list()
    
  for i in range(no):
    temp = generated_data_curr[i,:ori_time[i],:]
    generated_data.append(temp)
        
  # Renormalization
  generated_data = generated_data * max_val
  generated_data = generated_data + min_val
  end_gen=time.time()
  gen_time=end_gen-start_gen
  print(f">>>>>>>>>>>>>>>>>>>>>>模型生成时间:{gen_time :.4f} s<<<<<<<<<<<<<<<<<<")
  with open(result_path, 'a') as file:  # 使用 'w' 模式，每次都会覆盖文件
    file.write(f"模型生成时间:{gen_time :.4f} s")  # 每个结果之间换行
  return generated_data
