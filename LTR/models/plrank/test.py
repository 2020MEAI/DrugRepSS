import argparse
import numpy as np
import time
import tensorflow as tf
import json
import torch
import algorithms.PLRank as plr
import algorithms.pairwise as pw
import algorithms.lambdaloss as ll
import algorithms.tensorflowloss as tfl
import utils.dataset as dataset
import utils.nnmodel as nn
import utils.evaluate as evl
import utils.ranking as rnk

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str, default="local_output/model.pth",
                    help="Path to output model.")
parser.add_argument("--fold_id", type=int,
                    help="Fold number to select, modulo operator is applied to stay in range.",
                    default=1)
parser.add_argument("--dataset", type=str,
                    default="mydataset",
                    help="Name of dataset.")
parser.add_argument("--dataset_info_path", type=str,
                    default="example_datasets_info.txt",
                    help="Path to dataset info file.")
parser.add_argument("--cutoff", type=int,
                    help="Maximum number of items that can be displayed.",
                    default=10)
parser.add_argument("--num_samples", default='dynamic',
                    help="Number of samples for gradient estimation ('dynamic' applies the dynamic strategy).")
parser.add_argument("--num_eval_samples", type=int,
                    help="Number of samples for metric calculation in evaluation.",
                    default=100)
parser.add_argument("--loss", type=str, default='PL_rank_2',
                    help="Name of the loss to use (PL_rank_1/PL_rank_2/lambdaloss/pairwise/policygradient/placementpolicygradient).")
parser.add_argument("--timed", action='store_true',
                    help="Turns off evaluation so method can be timed.")
parser.add_argument("--vali", action='store_true',
                    help="Results calculated on the validation set.")

args = parser.parse_args()

cutoff = args.cutoff
num_samples = args.num_samples
num_eval_samples = args.num_eval_samples
timed_run = args.timed
validation_results = args.vali


if num_samples == 'dynamic':
    dynamic_samples = True


if args.dataset == 'mydataset':
    n_epochs = 100


data = dataset.get_dataset_from_json_info(
                  args.dataset,
                  args.dataset_info_path,
                )
fold_id = (args.fold_id-1)%data.num_folds()
data = data.get_data_folds()[fold_id]
data.read_data()
qids_test = []
for line in open('dataset/test.txt', 'r'):
    info = line[:line.find('#')].split()
    qid = info[1].split(':')[1]
    if qid not in qids_test:
        qids_test.append(qid)
print("数据读取完毕！！")

max_ranking_size = np.max((cutoff, data.max_query_size()))
metric_weights = 1./np.log2(np.arange(max_ranking_size) + 2)
train_labels = 2**data.train.label_vector-1
test_labels = 2**data.test.label_vector-1
ideal_train_metrics = evl.ideal_metrics(data.train, metric_weights, train_labels)
ideal_test_metrics = evl.ideal_metrics(data.test, metric_weights, test_labels)

model_params = {'hidden units': [32, 32],
                'learning_rate': 0.0001,}

model = nn.init_model(model_params)
optimizer = tf.keras.optimizers.SGD(learning_rate=model_params['learning_rate'])


#############################  训练模型
n_queries = data.train.num_queries()
if dynamic_samples:
  num_samples = 10
  float_num_samples = 10.
  add_per_step = 90./(n_queries*40.)
  max_num_samples = 100

for epoch_i in range(n_epochs):
  query_permutation = np.random.permutation(n_queries)
  for qid in query_permutation:
    q_labels =  data.train.query_values_from_vector(
                              qid, train_labels)
    q_feat = data.train.query_feat(qid)
    q_ideal_metric = ideal_train_metrics[qid]

    if q_ideal_metric != 0:
      q_metric_weights = metric_weights #/q_ideal_metric #uncomment for NDCG
      with tf.GradientTape() as tape:
        q_tf_scores = model(q_feat)
        # last_method_train_time = time.time()
        if args.loss == 'policygradient':
          loss = tfl.policy_gradient(
                                    q_metric_weights,
                                    q_labels,
                                    q_tf_scores,
                                    n_samples=num_samples
                                    )
          # method_train_time += time.time() - last_method_train_time
        elif args.loss == 'placementpolicygradient':
          loss = tfl.placement_policy_gradient(
                                    q_metric_weights,
                                    q_labels,
                                    q_tf_scores,
                                    n_samples=num_samples
                                    )
          # method_train_time += time.time() - last_method_train_time
        else:
          q_np_scores = q_tf_scores.numpy()[:,0]
          if args.loss == 'pairwise':
            doc_weights = pw.pairwise(q_labels,
                                      q_np_scores,
                                      )
          elif args.loss == 'lambdaloss':
            doc_weights = ll.lambdaloss(
                                      q_metric_weights,
                                      q_labels,
                                      q_np_scores,
                                      n_samples=num_samples
                                      )
          elif args.loss == 'PL_rank_1':
            doc_weights = plr.PL_rank_1(
                                      q_metric_weights,
                                      q_labels,
                                      q_np_scores,
                                      n_samples=num_samples)
          elif args.loss == 'PL_rank_2':
            doc_weights = plr.PL_rank_2(
                                      q_metric_weights,
                                      q_labels,
                                      q_np_scores,
                                      n_samples=num_samples)
          else:
            raise NotImplementedError('Unknown loss %s' % args.loss)
          # method_train_time += time.time() - last_method_train_time

          loss = -tf.reduce_sum(q_tf_scores[:,0] * doc_weights)

      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  print("第{}轮训练结束".format(epoch_i+1))


print("ok")

cutoff = metric_weights.size
scores = model(data.test.feature_matrix).numpy()[:, 0]
with open('output-gnncover/100epoch_PL_rank_3.txt', 'w') as f:
    for qid in range(data.test.num_queries()):
        q_scores = data.test.query_values_from_vector(qid, scores)
        q_labels = data.test.query_values_from_vector(qid, test_labels)
        ranking = rnk.cutoff_ranking(-q_scores, cutoff)
        ranking += data.test.doclist_ranges[qid]
        # 将 qid 和逗号分隔的 ranking 写入文件
        line = f"{qids_test[qid]} {','.join(map(str, ranking))}\n"
        f.write(line)
        print(qids_test[qid])
        print(ranking)



print("ok")