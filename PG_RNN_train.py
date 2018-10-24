
import json
import os
import time
import urllib
import zipfile
import h5py
# internal imports
import tensorflow.contrib.slim as slim
import numpy as np
import requests
import six
from six.moves import cStringIO as StringIO
import tensorflow as tf

import model as sketch_rnn_model
import utils
import pdb
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string(
    'data_dir',
    '/import/vision-datasets001/kl303/PG_data/PG_ndjson/fine_tuning1/',

    'The directory in which to find the dataset specified in model hparams. '
    'If data_dir starts with "http://" or "https://", the file will be fetched '
    'remotely.')
tf.app.flags.DEFINE_string(
    'log_root', './models',
    'Directory to store model checkpoints, tensorboard.')
tf.app.flags.DEFINE_boolean(
    'resume_training', False,
    'Set to true to load previous checkpoint')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'Pass in comma-separated key=value pairs such as '
    '\'save_every=40,decay_rate=0.99\' '
    '(no whitespace) to be read into the HParams object defined in model.py')

PRETRAINED_MODELS_URL = ('http://download.magenta.tensorflow.org/models/'
                         'sketch_rnn.zip')


def reset_graph():
  """Closes the current default session and resets the graph."""
  sess = tf.get_default_session()
  if sess:
    sess.close()
  tf.reset_default_graph()


def load_env(data_dir, model_dir):
  """Loads environment for inference mode, used in jupyter notebook."""
  model_params = sketch_rnn_model.get_default_hparams()
  with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
    model_params.parse_json(f.read())
  return load_dataset(data_dir, model_params, inference_mode=True)


def load_model(model_dir):
  """Loads model for inference mode, used in jupyter notebook."""
  model_params = sketch_rnn_model.get_default_hparams()
  with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
    model_params.parse_json(f.read())

  model_params.batch_size = 1  # only sample one at a time
  eval_model_params = sketch_rnn_model.copy_hparams(model_params)
  eval_model_params.use_input_dropout = 0
  eval_model_params.use_recurrent_dropout = 0
  eval_model_params.use_output_dropout = 0
  eval_model_params.is_training = 0
  sample_model_params = sketch_rnn_model.copy_hparams(eval_model_params)
  sample_model_params.max_seq_len = 1  # sample one point at a time
  return [model_params, eval_model_params, sample_model_params]


def download_pretrained_models(
    models_root_dir='./models',
    pretrained_models_url=PRETRAINED_MODELS_URL):
  """Download pretrained models to a temporary directory."""
  tf.gfile.MakeDirs(models_root_dir)
  zip_path = os.path.join(
      models_root_dir, os.path.basename(pretrained_models_url))
  if os.path.isfile(zip_path):
    tf.logging.info('%s already exists, using cached copy', zip_path)
  else:
    tf.logging.info('Downloading pretrained models from %s...',
                    pretrained_models_url)
    urllib.urlretrieve(pretrained_models_url, zip_path)
    tf.logging.info('Download complete.')
  tf.logging.info('Unzipping %s...', zip_path)
  with zipfile.ZipFile(zip_path) as models_zip:
    models_zip.extractall(models_root_dir)
  tf.logging.info('Unzipping complete.')



def load_dataset(data_dir, model_params, inference_mode=False):
  # aug_data_dir ='/import/vision-datasets/kl303/PG_data/svg_fine_tuning/Aug_data/'


  datasets = model_params.data_set
  model_params.data_set = datasets
  train_strokes = None
  valid_strokes = None
  eval_strokes = None

  for dataset in datasets:

    with open(data_dir+dataset+'.ndjson','r') as f:
      ori_data = json.load(f)
      train_stroke = ori_data['train_data'][:650]
      valid_stroke = ori_data['train_data'][650:700]
      eval_stroke = ori_data['train_data'][700:]

    if train_strokes is None:
      train_strokes = train_stroke
    else:
      train_strokes = np.concatenate((train_strokes, train_stroke))
    if valid_strokes is None:
      valid_strokes = valid_stroke
    else:
      valid_strokes = np.concatenate((valid_strokes, valid_stroke))
    if eval_strokes is None:
      eval_strokes = eval_stroke
    else:
      eval_strokes = np.concatenate((eval_strokes, eval_stroke))


  all_strokes = np.concatenate((train_strokes, valid_strokes, eval_strokes))
  #all_strokes = train_strokes
  num_points = 0
  for stroke in all_strokes:
    num_points += len(stroke)

  # calculate the max strokes we need.
  max_seq_len = utils.get_max_len(all_strokes)
  # overwrite the hps with this calculation.
  model_params.max_seq_len = max_seq_len

  tf.logging.info('model_params.max_seq_len %i.', model_params.max_seq_len)

  eval_model_params = sketch_rnn_model.copy_hparams(model_params)

  eval_model_params.use_input_dropout = 0
  eval_model_params.use_recurrent_dropout = 0
  eval_model_params.use_output_dropout = 0
  eval_model_params.is_training = 1

  if inference_mode:
    eval_model_params.batch_size = 1
    eval_model_params.is_training = 0

  sample_model_params = sketch_rnn_model.copy_hparams(eval_model_params)
  sample_model_params.batch_size = 1  # only sample one at a time
  sample_model_params.max_seq_len = 1  # sample one point at a time

  #pdb.set_trace()
  train_set = utils.DataLoader(
      train_strokes,
      model_params.batch_size,
      max_seq_length=model_params.max_seq_len,
      random_scale_factor=model_params.random_scale_factor,
      augment_stroke_prob=model_params.augment_stroke_prob)

  normalizing_scale_factor = train_set.calculate_normalizing_scale_factor()
  train_set.normalize(normalizing_scale_factor)

  valid_set = utils.DataLoader(
      valid_strokes,
      eval_model_params.batch_size,
      max_seq_length=eval_model_params.max_seq_len,
      random_scale_factor=0.0,
      augment_stroke_prob=0.0)
  valid_set.normalize(normalizing_scale_factor)
  #
  test_set = utils.DataLoader(
      eval_strokes,
      eval_model_params.batch_size,
      max_seq_length=eval_model_params.max_seq_len,
      random_scale_factor=0.0,
      augment_stroke_prob=0.0)
  test_set.normalize(normalizing_scale_factor)

  tf.logging.info('normalizing_scale_factor %4.4f.', normalizing_scale_factor)


  result = [train_set,valid_set,test_set, model_params, eval_model_params,sample_model_params]
  return result


def evaluate_model(sess, model, data_set):
  """Returns the average weighted cost, reconstruction cost and KL cost."""
  total_g_cost=0.0
  test_ac=0.0
  for batch in range(data_set.num_batches):
    unused_orig_x, x,labels,str_labels, s = data_set.get_batch(batch)
    feed = {model.input_data: x, model.sequence_lengths: s,model.labels:labels,model.str_labels:str_labels}
    (g_cost,ac) = sess.run([model.g_cost,model.accuracy], feed)

    total_g_cost += g_cost
    test_ac +=ac

  total_g_cost /= (data_set.num_batches)
  test_ac /= (data_set.num_batches)
  return (total_g_cost,test_ac)


def load_checkpoint(checkpoint_path,checkpoint_exclude_scopes):
  ckpt = tf.train.get_checkpoint_state(checkpoint_path)
  pretrain_model = ckpt.model_checkpoint_path
  print("load pretrained model from %s" % pretrain_model)

  exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

  variables_to_restore = []

  for var in tf.trainable_variables():
    excluded = False
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      print(var.name)
      variables_to_restore.append(var)
  return slim.assign_from_checkpoint_fn(pretrain_model, variables_to_restore)


def save_model(sess, model_save_path, global_step,saver):
  checkpoint_path = os.path.join(model_save_path, 'vector')
  tf.logging.info('saving model %s.', checkpoint_path)
  tf.logging.info('global_step %i.', global_step)
  saver.save(sess, checkpoint_path, global_step=global_step)

def train(sess, model, eval_model, train_set, valid_set, test_set,saver):
  """Train a sketch-rnn model."""
  # Setup summary writer.
  summary_writer = tf.summary.FileWriter(FLAGS.log_root)
  # Calculate trainable params.
  t_vars = tf.trainable_variables()
  count_t_vars = 0
  for var in t_vars:
    num_param = np.prod(var.get_shape().as_list())
    count_t_vars += num_param
    tf.logging.info('%s %s %i', var.name, str(var.get_shape()), num_param)
  tf.logging.info('Total trainable variables %i.', count_t_vars)
  model_summ = tf.summary.Summary()
  model_summ.value.add(
      tag='Num_Trainable_Params', simple_value=float(count_t_vars))
  summary_writer.add_summary(model_summ, 0)
  summary_writer.flush()

  # setup eval stats
  best_valid_cost = 100000  # set a large init value
  valid_cost = 0.0

  # main train loop

  hps = model.hps
  start = time.time()

  for _ in range(hps.num_steps):

    step = sess.run(model.global_step)

    curr_learning_rate = ((hps.learning_rate - hps.min_learning_rate) *
                          (hps.decay_rate)**(step/3) + hps.min_learning_rate)
    curr_kl_weight = (hps.kl_weight - (hps.kl_weight - hps.kl_weight_start) *
                      (hps.kl_decay_rate)**(step/3))

    _, x,labels,seg_labels, s,triplet_label = train_set.random_batch()
    feed = {
        model.input_data: x,
        model.sequence_lengths: s,
        model.lr: curr_learning_rate,
        model.labels:labels,
        model.str_labels:seg_labels,
        model.triplets:triplet_label
    }
    (triplet_loss,g_cost,train_accuracy, _, pre_labels,train_step, _) = sess.run([
        model.triplets_loss,model.g_cost,model.accuracy,model.final_state,model.out_pre_labels,
        model.global_step, model.train_op], feed)
    (triplet_loss,g_cost,train_accuracy, _, pre_labels,train_step, _) = sess.run([
        model.triplets_loss,model.g_cost,model.accuracy,model.final_state,model.out_pre_labels,
        model.global_step, model.train_op], feed)
    if step % 10 == 0 and step > 0:
    #if step % 1 == 0 and step > 0:
      end = time.time()
      time_taken = end - start


      g_summ = tf.summary.Summary()
      g_summ.value.add(tag='Train_group_Cost', simple_value=float(g_cost))
      lr_summ = tf.summary.Summary()
      lr_summ.value.add(
          tag='Learning_Rate', simple_value=float(curr_learning_rate))
      kl_weight_summ = tf.summary.Summary()
      kl_weight_summ.value.add(
          tag='KL_Weight', simple_value=float(curr_kl_weight))
      time_summ = tf.summary.Summary()
      time_summ.value.add(
          tag='Time_Taken_Train', simple_value=float(time_taken))
      accuracy_summ = tf.summary.Summary()
      accuracy_summ.value.add(
      tag='train_accuracy', simple_value=float(train_accuracy))
      output_format = ('step: %d, lr: %.6f, cost: %.4f,'
                       'train_time_taken: %.4f,train_accuracy: %.4f')
      output_values = (step, curr_learning_rate,  g_cost, time_taken,train_accuracy)
      output_log = output_format % output_values

      tf.logging.info(output_log)

      summary_writer.add_summary(g_summ, train_step)
      summary_writer.add_summary(lr_summ, train_step)
      summary_writer.add_summary(kl_weight_summ, train_step)
      summary_writer.add_summary(time_summ, train_step)
      summary_writer.flush()
      start = time.time()

    if step % hps.save_every == 0 and step > 0:
      (valid_g_cost,valid_ac) = evaluate_model(sess, eval_model, valid_set)
      valid_cost=valid_g_cost
      end = time.time()
      time_taken_valid = end - start
      start = time.time()

      valid_g_summ = tf.summary.Summary()
      valid_g_summ.value.add(
          tag='Valid_group_Cost', simple_value=float(valid_g_cost))
      valid_time_summ = tf.summary.Summary()
      valid_time_summ.value.add(
          tag='Time_Taken_Valid', simple_value=float(time_taken_valid))

      output_format = ('best_valid_cost: %0.4f, valid_g_cost: %.4f, valid_time_taken: %.4f,valid_ac: %.4f')
      output_values = (min(best_valid_cost, valid_g_cost), valid_g_cost, time_taken_valid,valid_ac)
      output_log = output_format % output_values

      tf.logging.info(output_log)

      summary_writer.add_summary(valid_g_summ, train_step)
      summary_writer.add_summary(valid_time_summ, train_step)
      summary_writer.flush()

      if valid_cost < best_valid_cost:
        best_valid_cost = valid_cost

        save_model(sess, FLAGS.log_root, step,saver)

        end = time.time()
        time_taken_save = end - start
        start = time.time()

        tf.logging.info('time_taken_save %4.4f.', time_taken_save)

        best_valid_cost_summ = tf.summary.Summary()
        best_valid_cost_summ.value.add(
            tag='Best_Valid_Cost', simple_value=float(best_valid_cost))

        summary_writer.add_summary(best_valid_cost_summ, train_step)
        summary_writer.flush()

        (eval_g_cost,eval_ac) = evaluate_model(sess, eval_model, test_set)

        end = time.time()
        time_taken_eval = end - start
        start = time.time()

        eval_g_summ = tf.summary.Summary()
        eval_g_summ.value.add(
            tag='Eval_group_Cost', simple_value=float(eval_g_cost))
        eval_accuracy_summ = tf.summary.Summary()
        eval_accuracy_summ.value.add(
          tag='eval_accuracy', simple_value=float(eval_ac))
        eval_time_summ = tf.summary.Summary()
        eval_time_summ.value.add(
            tag='Time_Taken_Eval', simple_value=float(time_taken_eval))

        output_format = ('eval_g_cost: %.4f, '
                         'eval_time_taken: %.4f,eval_accuracy: %.4f')
        output_values = (eval_g_cost, time_taken_eval,eval_ac)
        output_log = output_format % output_values

        tf.logging.info(output_log)

        summary_writer.add_summary(eval_g_summ, train_step)
        summary_writer.add_summary(eval_time_summ, train_step)
        summary_writer.flush()


def trainer(model_params,sess):
  """Train a sketch-rnn model."""
  np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

  tf.logging.info('PG-rnn')
  tf.logging.info('Hyperparams:')
  for key, val in six.iteritems(model_params.values()):
    tf.logging.info('%s = %s', key, str(val))
  tf.logging.info('Loading data files.')
  datasets = load_dataset(FLAGS.data_dir, model_params)

  train_set = datasets[0]
  valid_set = datasets[1]
  test_set = datasets[2]
  model_params = datasets[3]
  eval_model_params = datasets[4]

  model = sketch_rnn_model.Model(model_params)
  eval_model = sketch_rnn_model.Model(eval_model_params, reuse=True)
  
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(tf.global_variables())
  if FLAGS.resume_training:
    init_op = load_checkpoint(FLAGS.log_root,[])
    init_op(sess)

  # Write config file to json file.
  tf.gfile.MakeDirs(FLAGS.log_root)
  with tf.gfile.Open(
      os.path.join(FLAGS.log_root, 'model_config.json'), 'w') as f:
    json.dump(model_params.values(), f, indent=True)

  train(sess, model, eval_model, train_set,valid_set,test_set,saver)


def main(unused_argv):
  """Load model params, save config file and start trainer."""
  sess = tf.Session()
  default_model_params = sketch_rnn_model.get_default_hparams()
  trainer(default_model_params,sess)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
