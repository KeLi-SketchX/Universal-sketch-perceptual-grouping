
import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import KMeans
import math
import timeit
import logging
import os.path
import pdb
import os
import h5py
import json
import six
import model as sketch_rnn_model
import utils
import svgwrite
import scipy.io as scio
import pre_label2BSR
from copy import deepcopy
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'data_dir',
    '/import/vision-datasets001/kl303/PG_data/PG_ndjson/fine_tuning1/',
    'The directory in which to find the dataset specified in model hparams. '
    'If data_dir starts with "http://" or "https://", the file will be fetched '
    'remotely.')
tf.app.flags.DEFINE_string(
    'log_root', './models/',
    'Directory to store model checkpoints, tensorboard.')
tf.app.flags.DEFINE_boolean(
    'resume_training', True,
    'Set to true to load previous checkpoint')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'Pass in comma-separated key=value pairs such as '
    '\'save_every=40,decay_rate=0.99\' '
    '(no whitespace) to be read into the HParams object defined in model.py')

PRETRAINED_MODELS_URL = ('http://download.magenta.tensorflow.org/models/'
                         'sketch_rnn.zip')
def draw_strokes(data,group_labels, factor=1, svg_filename = '../sample.svg'):
    color_data = scio.loadmat('../colors.mat')
    color = color_data['colors']
    min_x, max_x, min_y, max_y = utils.get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
    lift_pen = 1
    abs_x = 25 - min_x
    abs_y = 25 - min_y
    p = "M%s,%s " % (abs_x, abs_y)
    real_stroke_flag = 0
    command = "m"
    begin_point = [abs_x,abs_y]
    end_point = [0,0]
    for i in xrange(len(data)):
        group_no = group_labels[i]
        current_color = color[np.mod(group_no, 10)]
        if (lift_pen == 1):
            # if real_stroke_flag:
            #     dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
            command = "m"
            x = float(data[i][0]) / factor
            y = float(data[i][1]) / factor
            abs_x += x
            abs_y += y
            lift_pen = data[i][2]

            real_stroke_flag = 0
        elif (command != "l"):
            command = "l"
            x = float(data[i][0]) / factor
            y = float(data[i][1]) / factor
            begin_point = [abs_x, abs_y]
            abs_x += x
            abs_y += y
            end_point = [abs_x, abs_y]
            lift_pen = data[i][2]
            dwg.add(dwg.line((begin_point[0], begin_point[1]), (end_point[0], end_point[1]),
                             stroke=svgwrite.rgb(current_color[0] * 100, current_color[1] * 100, current_color[2] * 100,
                                                 '%')))
            real_stroke_flag = 1
        else:
            command = ""
            x = float(data[i][0]) / factor
            y = float(data[i][1]) / factor
            begin_point = [abs_x, abs_y]
            abs_x += x
            abs_y += y
            end_point = [abs_x, abs_y]
            lift_pen = data[i][2]
            dwg.add(dwg.line((begin_point[0], begin_point[1]), (end_point[0], end_point[1]),
                             stroke=svgwrite.rgb(current_color[0] * 100, current_color[1] * 100, current_color[2] * 100,
                                                 '%')))
            real_stroke_flag = 1
    dwg.save()

def line_label2group_label(line_labels,str_labels_matrix):
    str_labels = str_labels_matrix[1,:]
    temp_i =1
    while np.sum(str_labels)==0:
        temp_i +=1
        str_labels= str_labels_matrix[temp_i,:]
    line_group_label = []
    gap_idx = np.where(str_labels==0)[0]
    str_num = len(gap_idx)
    line_idx = 0
    for str_idx,str_label in enumerate(str_labels):
        if str_label==0:
            line_label = line_labels[line_idx]
            line_group_label.append(line_label)
        else:
            line_label = line_labels[line_idx]
            line_group_label.append(line_label)
            line_idx+=1
    return line_group_label

def load_dataset(data_dir, model_params, inference_mode=False):
    #test_data_dir = '/import/vision-datasets001/kl303/PG_data/PG_ndjson/fine_tuning1/'
    test_data_dir = data_dir
    # test_data_dir = '../ECCV_Vrandom_noise/random_noise_test_data/'
    all_datasets = model_params.all_data_set
    test_datasets = model_params.test_data_set
    # model_params.data_set = datasets
    train_strokes = None
    valid_strokes = None
    eval_strokes = None
    testsss_strokes =None
    # for dataset in all_datasets:
    #     if dataset in test_datasets:
    #         continue

    for dataset in test_datasets:


        with open(test_data_dir + dataset + '.ndjson', 'r') as f:
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
    # all_strokes = train_strokes
    num_points = 0
    for stroke in all_strokes:
        num_points += len(stroke)
    max_seq_len = utils.get_max_len(all_strokes)

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

    train_set = utils.DataLoader(
        train_strokes,
        model_params.batch_size,
        max_seq_length=model_params.max_seq_len,
        random_scale_factor=model_params.random_scale_factor,
        augment_stroke_prob=model_params.augment_stroke_prob)

    # normalizing_scale_factor = train_set.calculate_normalizing_scale_factor()
    normalizing_scale_factor = 12.06541
    train_set.normalize(normalizing_scale_factor)

    valid_set = utils.DataLoader(
       valid_strokes,
       eval_model_params.batch_size,
       max_seq_length=eval_model_params.max_seq_len,
       random_scale_factor=0.0,
       augment_stroke_prob=0.0)
    valid_set.normalize(normalizing_scale_factor)

    test_set = utils.DataLoader(
       eval_strokes,
       eval_model_params.batch_size,
       max_seq_length=eval_model_params.max_seq_len,
       random_scale_factor=0.0,
       augment_stroke_prob=0.0)
    test_set.normalize(normalizing_scale_factor)

    tf.logging.info('normalizing_scale_factor %4.4f.', normalizing_scale_factor)

    result = [train_set, valid_set, test_set, model_params, eval_model_params, sample_model_params]

    return result


def get_init_fn(checkpoint_dir, checkpoint_exclude_scopes):
    #pdb.set_trace()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
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
    return slim.assign_from_checkpoint_fn(pretrain_model,variables_to_restore)

class PG_cluster_Rnn():


    Ks = 60  # the number of nearest neighbours of a sample
    Kc = 4  # the number of nearest clusters of a cluster
    a = 1.0
    l = 1.0 # lambda
    alpha = 0  # -0.2
    epochs = 5  # 20
    batch_size = 100
    gamma_tr = 2  # weight of positive pairs in weighted triplet loss.
    margin = 0.2  # margin for weighted triplet loss
    num_nsampling = 2  # number of negative samples for each positive pairs to construct triplet.
    gamma_lr = 0.0001  # gamma for inverse learning rate policy
    power_lr = 0.75  # power for inverse learning rate policy
    p = 0
    iter_cnn = 0

    def __init__(self, dataset, RC=True, updateCNN=True, eta=0.9):

        self.sess = tf.InteractiveSession()

        self.dataset = dataset
        self.RC = RC
        self.updateCNN = updateCNN
        self.eta = eta
        self.tic = timeit.default_timer()
        np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)
        self.model_params = sketch_rnn_model.get_default_hparams()
        if FLAGS.hparams:
            self.model_params.parse(FLAGS.hparams)

        tf.logging.info('sketch-rnn')
        tf.logging.info('Hyperparams:')
        for key, val in six.iteritems(self.model_params.values()):
            tf.logging.info('%s = %s', key, str(val))
        tf.logging.info('Loading data files.')

        datasets = load_dataset(FLAGS.data_dir, self.model_params)

        self.train_set = datasets[0]
        self.valid_set = datasets[1]
        self.test_set = datasets[2]
        model_params = datasets[3]
        eval_model_params = datasets[4]

        # self.train_set = datasets[0]
        # model_params = datasets[1]

        self.model = sketch_rnn_model.Model(model_params)
        self.eval_model = sketch_rnn_model.Model(eval_model_params, reuse=True)

        tf.gfile.MakeDirs(FLAGS.log_root)
        with tf.gfile.Open(
                os.path.join(FLAGS.log_root, 'model_config.json'), 'w') as f:
            json.dump(self.model_params.values(), f, indent=True)

        # set up logging to file - see previous section for more details
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename='./logfile/logfile.log',
                            filemode='w')
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
        self.logger = logging.getLogger('')
        self.Ns = 300

    def clusters_init(self, indices):
        # initialize labels for input data given knn indices
        labels = -np.ones(self.Ns, np.int)
        num_class = 0
        for i in range(self.Ns):
            pos = []
            cur_idx = i
            while labels[cur_idx] == -1:
                pos.append(cur_idx)
                neighbor = indices[cur_idx, 0]
                labels[cur_idx] = -2
                cur_idx = neighbor
                if np.size(pos) > 50:
                    break
            if labels[cur_idx] < 0:
                labels[cur_idx] = num_class
                num_class += 1
            for idx in pos:
                labels[idx] = labels[cur_idx]

        self.Nc = num_class
        # self.logger.info('%.2f s, Nc is %d...', timeit.default_timer() - self.tic, self.Nc)
        return labels

    def get_Dis(self, fea, k):
        # Calculate Dis
        # self.logger.info('%.2f s, Begin to fit neighbour graph', timeit.default_timer() - self.tic)
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(fea)
        # self.logger.info('%.2f s, Finished fitting, begin to calculate Dis', timeit.default_timer() - self.tic)
        sortedDis, indexDis = neigh.kneighbors()
        # self.logger.info('%.2f s, Finished the calculation of Dis', timeit.default_timer() - self.tic)
        # mul_values = np.matmul(fea, (fea.T))
        # np.fill_diagonal(mul_values, -1)
        # # cos_dis = []
        # sortedDis = []
        # indexDis = []
        # for idx,line_cos_dis in enumerate(mul_values):
        #     A_model = np.linalg.norm(fea[idx])
        #     B_models = np.linalg.norm(fea,axis=1)
        #     norm_cos_dis = line_cos_dis/(A_model*B_models)
        #     sort_dis = -np.sort(-norm_cos_dis)
        #     index_dis = np.argsort(-norm_cos_dis)
        #     sortedDis.append(list(sort_dis[:-1]))
        #     indexDis.append(list(index_dis[:-1]))
            # cos_dis.append(norm_cos_dis)


        return np.asarray(sortedDis), np.asarray(indexDis)

    def get_A(self, fea, sortedDis, indexDis, C):
        # Calculate W
        sortedDis = np.power(sortedDis, 2)
        sig2 = sortedDis.sum() / (self.Ks * self.Ns) * self.a
        XI = np.transpose(np.tile(range(self.Ns), (self.Ks, 1)))
        W = csr_matrix((np.exp(-sortedDis.flatten() * (1 / sig2)), (XI.flatten(), indexDis.flatten())),
                       shape=(self.Ns, self.Ns)).toarray()
        # self.logger.info('%.2f s, Finished the calculation of W, sigma:%f', timeit.default_timer() - self.tic, np.sqrt(sig2))

        # Calculate A
        asymA = np.zeros((self.Nc, self.Nc))
        A = np.zeros((self.Nc, self.Nc))
        for i in range(self.Nc):
            # self.logger.info('%.2f s, Calculating A..., i: %d', timeit.default_timer() - self.tic, i)
            for j in range(i):
                if np.size(C[j], 0) != 0:
                    # asymA[i, j] = np.dot(np.sum(W[C[i], :][:, C[j]], 0), np.sum(W[C[j], :][:, C[i]], 1))  # A(Ci -> Cj)
                    asymA[i, j] = np.dot(np.sum(W[C[i], :][:, C[j]], 0), np.sum(W[C[j], :][:, C[i]], 1)) / math.pow(
                        np.size(C[j], 0), 2)  # A(Ci -> Cj)
                if np.size(C[i], 0) != 0:
                    # asymA[j, i] = np.dot(np.sum(W[C[j], :][:, C[i]], 0), np.sum(W[C[i], :][:, C[j]], 1))  # A(Cj -> Ci)
                    asymA[j, i] = np.dot(np.sum(W[C[j], :][:, C[i]], 0), np.sum(W[C[i], :][:, C[j]], 1)) / math.pow(
                        np.size(C[i], 0), 2)  # A(Cj -> Ci)
                # A[i, j] = asymA[i, j]/math.pow(np.size(C[j], 0), 2) + asymA[j, i]/ math.pow(np.size(C[i], 0), 2)
                A[i, j] = asymA[i, j] + asymA[j, i]
                A[j, i] = A[i, j]

        # Assert whether there are some self-contained clusters
        num_fea = np.size(fea, 1)
        if self.Nc > 20 * self.K:
            asymA_sum_row = np.sum(asymA, 0)
            asymA_sum_col = np.sum(asymA, 1)

            X_clusters = np.zeros((self.Nc, num_fea))
            for i in range(self.Nc):
                X_clusters[i, :] = np.mean(fea[C[i], :], 0)
            neigh = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(X_clusters)
            # self.logger.info('%.2f s, Fit the nearest neighbor graoh of clusters', timeit.default_timer() - self.tic)

            i = 0
            while i < self.Nc:
                # Find the cluster ids whose affinities are both 0
                if asymA_sum_row[i] == 0 and asymA_sum_col[i] == 0:
                    indices = neigh.kneighbors(X_clusters[i, :].reshape(1, -1), return_distance=False)
                    for j in indices[0]:
                        if i != j:
                            break
                    # self.logger.info('%.2f s, merge self-contained cluster %d with cluster %d', timeit.default_timer() - self.tic, i, j)
                    minIndex1 = min(i, j)
                    minIndex2 = max(i, j)
                    A, asymA, C = self.merge_cluster(A, asymA, C, minIndex1, minIndex2)
                    asymA_sum_row = np.sum(asymA, 0)
                    asymA_sum_col = np.sum(asymA, 1)

                    # update X_clusters
                    X_clusters[minIndex1, :] = np.mean(fea[C[minIndex1], :], 0)
                    X_clusters[minIndex2, :] = X_clusters[self.Nc, :]
                    X_clusters = X_clusters[0 : self.Nc, :]
                    neigh = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(X_clusters)
                else:
                    i += 1

        # self.logger.info('%.2f s, Calculation of A based on clusters is completed', timeit.default_timer() - self.tic)

        return A, asymA, C

    def find_closest_clusters(self, A):
        # Find two clusters with the smallest loss
        np.fill_diagonal(A, - float('Inf'))
        if self.Kc < len(A)/2:
            K = len(A)/2
        elif len(A)>6 and len(A)<15:
            K = 4
        else:
            K = len(A) -1
        indexA = A.argsort(axis=1)[:, ::-1][:, 0: K]
        sortedA = np.sort(A, axis=1)[:, ::-1][:, 0: K]

        minLoss = float('Inf')
        minIndex1 = -1
        minIndex2 = -2
        for i in range(self.Nc):
            loss = - (1 + self.l) * sortedA[i, 0] + sum(sortedA[i, 1: K]) * self.l / (K - 1)
            if loss < minLoss and i != indexA[i, 0]:
                minLoss = loss
                minIndex1 = min(i, indexA[i, 0])
                minIndex2 = max(i, indexA[i, 0])

        # self.logger.info('%.2f s, number of clusters: %d, merge cluster %d and %d, loss: %f', timeit.default_timer() - self.tic, self.Nc, minIndex1, minIndex2, minLoss)
        return minIndex1, minIndex2,minLoss

    def merge_cluster(self, A, asymA, C, minIndex1, minIndex2):

        # Merge
        cluster1 = C[minIndex1]
        cluster2 = C[minIndex2]
        new_cluster = cluster1 + cluster2

        # Update the merged cluster and its affinity
        C[minIndex1] = new_cluster
        asymA[minIndex1, 0: self.Nc] = asymA[minIndex1, 0: self.Nc] + asymA[minIndex2, 0: self.Nc]
        len1 = np.size(cluster1, 0)
        len2 = np.size(cluster2, 0)
        # asymA[0: self.Nc, minIndex1] = (asymA[0: self.Nc, minIndex1] * len1 + asymA[0: self.Nc, minIndex2] * len2) / (len1 + len2)
        asymA[0 : self.Nc, minIndex1] = asymA[0 : self.Nc, minIndex1] * (1 + self.alpha) * math.pow(len1, 2) / math.pow(len1 + len2, 2)\
                + asymA[0 : self.Nc, minIndex2] *(1 + self.alpha) * math.pow(len2, 2) / math.pow(len1 + len2, 2)
        asymA[minIndex1, minIndex1] = 0

        A[minIndex1, :] = asymA[minIndex1, :] + asymA[:, minIndex1]
        A[:, minIndex1] = A[minIndex1, :]

        # Replace the second cluster to be merged with the last cluster of the cluster array
        if (minIndex2 != self.Nc-1):
            C[minIndex2] = C[-1]
            asymA[0: self.Nc, minIndex2] = asymA[0: self.Nc, self.Nc - 1]
            asymA[minIndex2, 0: self.Nc] = asymA[self.Nc - 1, 0: self.Nc]
            asymA[minIndex2, minIndex2] = 0
            A[0: self.Nc, minIndex2] = A[0: self.Nc, self.Nc - 1]
            A[minIndex2, 0: self.Nc] = A[self.Nc - 1, 0: self.Nc]
            A[minIndex2, minIndex2] = 0

        # Remove the last cluster
        C.pop()
        asymA = asymA[0 : self.Nc - 1, 0 : self.Nc - 1]
        A = A[0 : self.Nc - 1, 0 : self.Nc - 1]
        self.Nc -= 1

        return A, asymA, C

    def get_C(self, labels):
        num_sam = np.size(labels)
        labels_from_one = np.zeros(num_sam, np.int)

        idx_sorted = labels.argsort(0)
        labels_sorted = labels[idx_sorted]
        nclusters = 0
        label = -1
        for i in range(num_sam):
            if labels_sorted[i] != label:
                label = labels_sorted[i]
                nclusters += 1
            labels_from_one[idx_sorted[i]] = nclusters

        C = []
        for i in range(nclusters):
            C.append([])
        for i in range(num_sam):
            C[labels_from_one[i] - 1].append(i)

        return C

    def get_labels(self, C):
        # Generate sample labels

        labels = np.zeros(self.Ns, np.int)
        for i in range(len(C)):
            labels[C[i]] = i

        return labels


    def cluster_model(self,):
        hps = self.model.hps

        self.y = tf.placeholder(tf.float32, shape=[None, 1])
        self.real_line_idx = tf.placeholder(tf.int32, shape=[None, 1])
        self.group_matrix = tf.squeeze(self.model.out_pre_labels)
        self.real_group_matrix = tf.gather(self.group_matrix,self.real_line_idx)

        total_loss = self.model.cost
        optimizer = tf.train.AdamOptimizer(self.model.lr)
        gvs = optimizer.compute_gradients(total_loss)
        g = hps.grad_clip
        for grad, var in gvs:
            tf.clip_by_value(grad, -g, g)
        capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(
            capped_gvs, global_step=self.model.global_step, name='train_step')

    def train(self, group_labels,x, labels, str_labels, s,step,real_line_idx):
        hps = self.model.hps
        self.get_triplet(group_labels)
        real_line_idx = np.reshape(real_line_idx,[-1,1])
        group_labels=np.reshape(group_labels,[-1,1])

        curr_learning_rate = ((hps.learning_rate - hps.min_learning_rate) *
                              (hps.decay_rate) ** step + hps.min_learning_rate)
        feed = {
            self.model.input_data: x,
            self.model.sequence_lengths: s,
            self.model.labels: labels,
            self.model.str_labels: str_labels,
            self.model.lr: curr_learning_rate,
            self.y: group_labels,
            self.real_line_idx:real_line_idx
        }
        epoch_triplet_loss = 0
        epoch_accuracy = 0
        for e in range(self.epochs):
            anc,triplet_loss,accuracy,_,real_group_matrix=self.sess.run([self.anc_idx,self.loss,self.model.accuracy,self.train_op,self.real_group_matrix], feed)
            epoch_triplet_loss+=triplet_loss
            epoch_accuracy+=accuracy
        epoch_triplet_loss /=self.epochs
        epoch_accuracy /=self.epochs
        self.logger.info('step:%.2f s, triplet loss: %f,learning_rate:%f,accuracy: %f',step, triplet_loss,curr_learning_rate,accuracy)
        feed = {
            self.model.input_data: x,
            self.model.sequence_lengths: s,
            self.model.labels: labels,
            self.model.str_labels: str_labels
        }
        pre_labels = self.sess.run([self.model.out_pre_labels], feed)
        pre_labels = np.asarray(pre_labels[0][0][0])
        return pre_labels

    def recurrent_process(self,updateCNN,x, labels, str_labels, s,step):

        self.K = len(np.unique(self.gnd))
        feed = {
            self.model.sequence_lengths: s,
            self.model.input_data: x,
            self.model.labels: labels,
            self.model.str_labels: str_labels
        }

        group_matrix = self.model.out_pre_labels
        pre_labels,accuracy = self.sess.run([group_matrix,self.model.accuracy], feed)
        pre_labels = np.asarray(pre_labels[0][0])
        if len(pre_labels.shape)==1:
            pre_labels = np.reshape(pre_labels,[s[0],s[0]])
        str_label = str_labels[0,:s[0],:s[0]]

        real_line_index = np.where(str_label[:, 1] == 1)[0]
        real_line_pre_labels = np.take(pre_labels,real_line_index,axis=0)
        real_line_pre_labels = np.take(real_line_pre_labels, real_line_index, axis=1)
        self.gnd = np.take(self.gnd,real_line_index)
        self.Ns = len(real_line_pre_labels)


        if self.Ks>=len(real_line_pre_labels):
            self.Ks=len(real_line_pre_labels)-2
        #pdb.set_trace()
        sortedDis, indexDis = self.get_Dis(real_line_pre_labels, self.Ks)
        group_labels = self.clusters_init(indexDis)
        C = self.get_C(group_labels)
        A, asymA, C = self.get_A(real_line_pre_labels, sortedDis, indexDis, C)
        total_minLoss = {}
        total_C={}
        # index1, index2, minLoss = self.find_closest_clusters(A)
        minLoss =0
        while self.Nc > 10:
            # if self.Nc < self.Ks*2 and self.Nc>20:
            #     self.Ks =self.Nc/2
            #     sortedDis, indexDis = self.get_Dis(real_line_pre_labels, self.Ks)
            #     A, asymA, C = self.get_A(real_line_pre_labels, sortedDis, indexDis, C)
            index1, index2,minLoss = self.find_closest_clusters(A)
            A, asymA, C = self.merge_cluster(A, asymA, C, index1, index2)

        # t = 0
        # ts = 0
        # Np = np.ceil(self.eta * self.Nc)
        # while (self.Nc > 15) and (minLoss!=-float('Inf')) :
        # total_minLoss = []
        total_minLoss = np.zeros(20)
        total_minLoss[0] = 10000
        while (self.Nc>3) and (minLoss<-0.08) and (~np.isinf(minLoss)):
        #     t += 1
        #     print self.Nc
        # while (self.Nc > 2):
            index1, index2,minLoss = self.find_closest_clusters(A)
            A, asymA, C = self.merge_cluster(A, asymA, C, index1, index2)
            total_C[str(self.Nc)] = deepcopy(C)
            total_minLoss[self.Nc] = minLoss


        # for loss_idx in range(2,19):
        #     minloss = total_minLoss[loss_idx]
        #     next_minloss = total_minLoss[loss_idx+1]
        #     if minloss< -0.05 or (next_minloss< -1 and minloss>-1 and minloss < 0):
        #         # pre_minloss_offset = total_minLoss[loss_idx-1]-minloss
        #         next_minloss_offset = minloss - next_minloss
        #         rate = next_minloss_offset/abs(minloss)
        #         if rate > 2 or (minloss< -0.5 and rate >1) :
        #
        #             # if minloss <-1:
        #             #     print loss_idx
        #             C = total_C[str(loss_idx)]
        #             break
        # if loss_idx>5:
        #     for loss_idx in range(3, 19):
        #         minloss = total_minLoss[loss_idx]
        #         if minloss<-0.08:
        #             C = total_C[str(loss_idx)]
        #             break
        return C,total_C,total_minLoss

    # def linelabel2grouplabel(self,line_labels,str_label):

    def test_k_mean_cluster(self,x, labels, str_labels, s,step):
        self.K = len(np.unique(self.gnd))
        feed = {
            self.model.sequence_lengths: s,
            self.model.input_data: x,
            self.model.labels: labels,
            self.model.str_labels: str_labels
        }

        group_matrix = self.model.out_pre_labels
        pre_labels = self.sess.run([group_matrix], feed)

        pre_labels = np.asarray(pre_labels[0][0][0])
        str_label = str_labels[0, :s[0], :s[0]]

        real_line_index = np.where(str_label[:, 1] == 1)[0]
        real_line_pre_labels = np.take(pre_labels, real_line_index, axis=0)
        self.K = len(np.unique(self.gnd))
        kmeans=KMeans(self.K).fit(real_line_pre_labels)
        y=kmeans.labels_
        return y

    def run(self):
        hps = self.model_params

        self.cluster_model()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())
        if FLAGS.resume_training:
            init_op = get_init_fn(FLAGS.log_root, [])
            init_op(self.sess)

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
        summary_writer = tf.summary.FileWriter(FLAGS.log_root)
        summary_writer.add_summary(model_summ, 0)
        summary_writer.flush()

        # pre_labels_path = FLAGS.log_root+'/pre_group_labels.h5'
        # str_labels_path = FLAGS.log_root+'/str_labels.h5'
        # str_labels_hf = h5py.File(str_labels_path,'w')
        # pre_labels_hf = h5py.File(pre_labels_path,'w')
        step =0
        test_accuracy =0
        #pre_group_labels =[]

        # right_pre_labels=h5py.File(FLAGS.log_root+'/pre_group_labels22500.h5','r')
        # right_str_labels = h5py.File(FLAGS.log_root + '/str_labels22500.h5', 'r')
        # keys = right_pre_labels.keys()
        eval_strokes = None

        #test_data_dir = '/import/vision-datasets001/kl303/PG_data/PG_ndjson/fine_tuning1/'
        test_data_dir = FLAGS.data_dir
        #all_datasets = self.model_params.all_data_set
        test_datasetes = self.model_params.test_data_set
        datasetes = test_datasetes
        out_put_file = './pre_label/DB/'
        if os.path.exists(out_put_file)==False:
            os.mkdir(out_put_file)
        for dataset in datasetes:
            with open(test_data_dir + dataset + '.ndjson', 'r') as f:
                ori_data = json.load(f)
                eval_stroke = ori_data['train_data'][700:]
            if eval_strokes is None:
                eval_strokes = eval_stroke
            else:
                eval_strokes = np.concatenate((eval_strokes, eval_stroke))
        # test_accuracymax(score_vector(1,:));
        use_set = self.test_set
        average_group = 0.0

        for test_idx in range(len(use_set.strokes)):
            _, x, labels, str_labels, s,stroke_group_label = use_set.get_batch(test_idx)
            self.gnd = stroke_group_label[0]
            C,C_dict,min_loss = self.recurrent_process(False, x, labels, str_labels, s,step)
            sketch_pre_line_labels = self.get_labels(C)

            average_group+=len(C)
            category_idx = test_idx / 100
            print test_idx#,len(C)
            #
            category = datasetes[category_idx]
            if os.path.exists(out_put_file + category) == False:
                os.mkdir(out_put_file + category)
            test_file_name = '/import/vision-datasets/kl303/PycharmProjects/BSR/bench/PG_data/test_file/' + category + '.txt'
            test_f = open(test_file_name, 'r')
            lines = test_f.readlines()
            line = lines[np.mod(test_idx, 100)].strip()
            test_f.close()
         #   pdb.set_trace()
            mat_file_name = out_put_file + category + '/' + line[:-4] + '.mat'
            pre_label2BSR.draw_sketch_with_strokes(eval_strokes[test_idx], sketch_pre_line_labels, mat_file_name)
            #print str(len(eval_strokes[test_idx]))+' '+str(len(sketch_pre_line_labels))
            # if os.path.exists(out_put_file + category + '/' + line[:-4]) == False:
            #     os.mkdir(out_put_file + category + '/' + line[:-4])
            # for c_key in C_dict.keys():
            #     C=C_dict[c_key]
            #     sketch_pre_line_labels = self.get_labels(C)
            #     sketch_pre_group_labels = line_label2group_label(sketch_pre_line_labels, str_labels[0, :s[0], :s[0]])
            #     # mat_file_name = out_put_file + category + '/' + line[:-4] +'/'+c_key+ '.mat'
            #     if os.path.exists('./visual_hard/' + str(test_idx)) == False:
            #         os.mkdir('./visual_hard/'+str(test_idx))
            #     svg_name = './visual_hard/' + str(test_idx)+'/'+c_key + '.svg'
            #     draw_strokes(eval_strokes[test_idx], sketch_pre_group_labels, 1, svg_name)
                # pre_label2BSR.draw_sketch_with_strokes(eval_strokes[test_idx],sketch_pre_line_labels,mat_file_name)

        # print 'average group number: ',average_group/len(use_set.strokes)

            # print test_idx,len(C),self.K
            # for key in C_dict.keys():
            #     C = C_dict[key]
            #     sketch_pre_line_labels = self.get_labels(C)
            #     sketch_pre_group_labels = line_label2group_label(sketch_pre_line_labels, str_labels[0, :s[0], :s[0]])
            #     if os.path.exists('./visual20/'+str(test_idx))==False:
            #         os.mkdir('./visual20/'+str(test_idx))
            #     svg_name = './visual20/' + str(test_idx)+'/'+key + '.svg'
            #     draw_strokes(eval_strokes[test_idx], sketch_pre_group_labels, 1, svg_name)
            # sketch_pre_line_labels = self.test_k_mean_cluster(x, labels, str_labels, s,step)
            # sketch_pre_group_labels = line_label2group_label(sketch_pre_line_labels, str_labels[0,:s[0],:s[0]])
            # svg_name = './visualfold1/'+str(test_idx)+'.svg'

            # str_labels_hf.create_dataset(str(test_idx),data=str_labels[0,:s[0],:s[0]])
            # pre_labels_hf.create_dataset(str(test_idx),data=sketch_pre_line_labels)
            # print('step:', test_idx,s[0],len(sketch_pre_group_labels),len(eval_strokes[test_idx]))

        # str_labels_hf.close()
        # pre_labels_hf.close()
        # print test_accuracy/len(use_set.strokes)


if __name__ == '__main__':
    PG_cluster_Rnn('umist', RC=True, updateCNN=False, eta=0.2).run()
