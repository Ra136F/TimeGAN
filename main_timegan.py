"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

main_timegan.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import warnings
import pandas as pd

from TimeGAN.DTW import compute_dtw_distance
from utils import extract_time, random_generator, calculate_rmse, calculate_mape, plottp, dtw_distance, \
    calculate_metrics

warnings.filterwarnings("ignore")

# 1. TimeGAN model
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation, restore_data, real_data_loading2
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization
import os
import tensorflow as tf


def main (args):
  """Main function for timeGAN experiments.
  
  Args:
    - data_name: sine, stock, or energy
    - seq_len: sequence length
    - Network parameters (should be optimized for different datasets)
      - module: gru, lstm, or lstmLN
      - hidden_dim: hidden dimensions
      - num_layer: number of layers
      - iteration: number of training iterations
      - batch_size: the number of samples in each batch
    - metric_iteration: number of iterations for metric computation
  
  Returns:
    - ori_data: original data
    - generated_data: generated synthetic data
    - metric_results: discriminative and predictive scores
  """
  ## Data loading
  ori_data=[]
  if args.data_name !='sine':
    ori_data,min,max = real_data_loading(args.data_name, args.seq_len,args.target,args.feature)
  elif args.data_name == 'sine':
    # Set number of samples and its dimensions
    no, dim = 10000, 5
    ori_data = sine_data_generation(no, args.seq_len, dim)
    
  print(args.data_name + ' dataset is ready.')
  print(f'数据集记录:{len(ori_data)}')
    
  ## Synthetic data generation by TimeGAN
  # Set newtork parameters
  parameters = dict()  
  parameters['module'] = args.module
  parameters['hidden_dim'] = args.hidden_dim
  parameters['num_layer'] = args.num_layer
  parameters['iterations'] = args.iteration
  parameters['batch_size'] = args.batch_size

  #加载模型
  save_name='data-{}'.format(args.data_name)
  model_path=os.path.join(args.model_path, save_name)
  result_path=os.path.join('./result/',save_name)
  if not os.path.exists(model_path):
      os.makedirs(model_path)
  if not os.path.exists(result_path):
      os.makedirs(result_path)
  model_path=model_path+'/'+args.model_name
  # if os.path.exists(model_path + ".meta"):
  #     print(f"Model found at {model_path}, loading the model.")
  #     load_model = True
  # else:
  #     print(f"No model found at {model_path}, starting training from scratch.")
  #     load_model = False
  generated_data,train_time,gen_time = timegan(ori_data, parameters,model_path,args.train,result_path)


  # generated_data = timegan(ori_data, parameters)
  print('Finish Synthetic Data Generation')
  
  # Performance metrics
  # Output initialization
  # metric_results = dict()
  #
  # # 1. Discriminative Score
  # discriminative_score = list()
  # for _ in range(args.metric_iteration):
  #   temp_disc = discriminative_score_metrics(ori_data, generated_data)
  #   discriminative_score.append(temp_disc)
  #
  # metric_results['discriminative'] = np.mean(discriminative_score)
  #
  # # 2. Predictive score
  # predictive_score = list()
  # for tt in range(args.metric_iteration):
  #   temp_pred = predictive_score_metrics(ori_data, generated_data)
  #   predictive_score.append(temp_pred)
  #
  # metric_results['predictive'] = np.mean(predictive_score)

  # 3. Visualization (PCA and tSNE)
  # visualization(ori_data, generated_data, 'pca')
  # visualization(ori_data, generated_data, 'tsne')

  ## Print discriminative and predictive scores
  # print(metric_results)

  return ori_data, generated_data,train_time,gen_time

def generate_synthetic_data(sess, ori_data, parameters):
    """Generate synthetic data using the trained TimeGAN model."""
    ori_time, max_seq_len = extract_time(ori_data)
    no, seq_len, dim = np.asarray(ori_data).shape
    z_dim = parameters['hidden_dim']
    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
    graph = tf.compat.v1.get_default_graph()
    X_hat = graph.get_tensor_by_name("X_hat:0")
    T = graph.get_tensor_by_name("myinput_t:0")
    Z = graph.get_tensor_by_name("myinput_z:0")
    generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, T: ori_time})
    generated_data = [
        generated_data_curr[i, :ori_time[i], :] for i in range(len(ori_data))
    ]
    return generated_data



if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['sine','stock','energy','Walmart','train'],
      default='energy',
      type=str)
  parser.add_argument(
      '--seq_len',
      help='sequence length',
      default=24,
      type=int)
  parser.add_argument(
      '--module',
      choices=['gru','lstm','lstmLN'],
      default='gru',
      type=str)
  parser.add_argument(
      '--hidden_dim',
      help='hidden state dimensions (should be optimized)',
      default=24,
      type=int)
  parser.add_argument(
      '--num_layer',
      help='number of layers (should be optimized)',
      default=3,
      type=int)
  parser.add_argument(
      '--iteration',
      help='Training iterations (should be optimized)',
      default=5000,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch (should be optimized)',
      default=128,
      type=int)
  parser.add_argument(
      '--metric_iteration',
      help='iterations of the metric computation',
      default=10,
      type=int)
  parser.add_argument(
      '--train',
      default=True,
      type=bool)
  parser.add_argument(
      '--model_path',
      default='./save_model/',
      type=str)
  parser.add_argument(
      '--model_name',
      default='model',
      type=str)
  parser.add_argument(
      '--target',
      default='Appliances',
      type=str)
  parser.add_argument('-feature', type=str, default='S', help='[S, MS],单元预测单元,多元预测单元')
  
  args = parser.parse_args() 
  
  # Calls main function  
  ori_data, generated_data,train_time,gen_time = main(args)
  #重新读入真实数据
  true_data, min, max = real_data_loading2(args.data_name, args.seq_len, args.target, args.feature)
  true_data = np.array(true_data)
  #三维转二维数组
  gen_data=restore_data(generated_data,args.seq_len,min,max)
  true_data = restore_data(true_data, args.seq_len,min,max)
  # real_y_true_mask = (1 - (true_data == 0))
  # rmse=calculate_rmse(true_data, gen_data)
  # mape=calculate_mape(true_data, gen_data, real_y_true_mask)
  #计算指标
  smape, rmse, mape = calculate_metrics(true_data, gen_data)
  print(f"RMSE: {rmse}")
  print(f"MAPE: {mape}")
  print(f'smape:{smape}')
  # 绘图
  plottp(true_data,gen_data,args.data_name)
  #计算DTW
  distance=dtw_distance(true_data,gen_data)
  print(f"DTW 距离: {distance}")
  #保存训练与生成时间
  time_path='./result'+'data-{}'.format(args.data_name)+'/'+'time-{}.txt'.format(args.iteration)
  with open(time_path, 'w') as file:
      file.write(f"模型已保存,用时:{train_time / 60:.4f} min\n")
      file.write(f"模型生成时间:{gen_time :.4f} s\n")

