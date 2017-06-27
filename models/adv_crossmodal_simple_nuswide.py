from __future__ import print_function
import os, time, cPickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from random import shuffle
import sklearn.preprocessing
from base_model import BaseModel, BaseModelParams, BaseDataIter
import utils
from flip_gradient import flip_gradient

def readtxt(filename):
    all_filename = []
    f = open(filename)
    line = f.readline()
    while line:
        line_ = line.strip('\n')
        all_filename.append(line_)
        line = f.readline()
    f.close()
    return all_filename

class DataIter(BaseDataIter):
    def __init__(self, batch_size):
        BaseDataIter.__init__(self, batch_size)
        self.num_train_batch = 0
        self.num_test_batch = 0

        with open('./data/nuswide/img_train_id_feats.pkl', 'rb') as f:
            self.train_img_feats = cPickle.load(f)
        with open('./data/nuswide/train_id_bow.pkl', 'rb') as f:
            self.train_txt_vecs = cPickle.load(f)
        with open('./data/nuswide/train_id_label_map.pkl', 'rb') as f:
            self.train_labels = cPickle.load(f)
        with open('./data/nuswide/img_test_id_feats.pkl', 'rb') as f:
            self.test_img_feats = cPickle.load(f)
        with open('./data/nuswide/test_id_bow.pkl', 'rb') as f:
            self.test_txt_vecs = cPickle.load(f)
        with open('./data/nuswide/test_id_label_map.pkl', 'rb') as f:
            self.test_labels = cPickle.load(f)
        with open('data/nuswide/train_ids.pkl', 'rb') as f:
            self.train_ids = cPickle.load(f)
        with open('data/nuswide/test_ids.pkl', 'rb') as f:
            self.test_ids = cPickle.load(f)  
        with open('data/nuswide/train_id_label_single.pkl', 'rb') as f:
            self.train_labels_single = cPickle.load(f)
        with open('data/nuswide/test_id_label_single.pkl', 'rb') as f:
            self.test_labels_single = cPickle.load(f)              
                
        np.random.shuffle(self.train_ids)
        np.random.shuffle(self.test_ids)          
        self.num_train_batch = len(self.train_ids) / self.batch_size
        self.num_test_batch = len(self.test_ids) / self.batch_size


    def train_data(self):
        for i in range(self.num_train_batch):
            batch_img_ids = self.train_ids[i*self.batch_size : (i+1)*self.batch_size]
            batch_img_feats = [self.train_img_feats[n] for n in batch_img_ids]
            batch_txt_vecs = [self.train_txt_vecs[n] for n in batch_img_ids]
            batch_labels = [self.train_labels[n] for n in batch_img_ids]
            batch_labels_single = np.array([self.train_labels_single[n] for n in batch_img_ids])
            yield batch_img_feats, batch_txt_vecs, batch_labels, batch_labels_single, i

    def test_data(self):
        for i in range(self.num_test_batch):
            batch_img_ids = self.test_ids[i*self.batch_size : (i+1)*self.batch_size]
            batch_img_feats = [self.test_img_feats[n] for n in batch_img_ids]
            batch_txt_vecs = [self.test_txt_vecs[n] for n in batch_img_ids]
            batch_labels = [self.test_labels[n] for n in batch_img_ids]
            batch_labels_single = [self.test_labels_single[n] for n in batch_img_ids]
            yield batch_img_feats, batch_txt_vecs, batch_labels, batch_labels_single, i


class ModelParams(BaseModelParams):
    def __init__(self):
        BaseModelParams.__init__(self)

        self.batch_size = 64
       
        self.visual_feat_dim = 4096
        #self.word_vec_dim = 200
        self.word_vec_dim = 1000
        self.lr_emb = 0.0001
        self.lr_domain = 0.0001
        self.top_k = 50
        self.semantic_emb_dim = 40
        self.dataset_name = 'nuswide'
        self.model_name = 'adv_semantic_zsl'
        self.model_dir = 'adv_semantic_zsl_%d_%d_%d' % (self.visual_feat_dim, self.word_vec_dim, self.semantic_emb_dim)

        self.checkpoint_dir = 'checkpoint'
        self.sample_dir = 'samples'
        self.dataset_dir = './data'
        self.log_dir = 'logs'

    def update(self):
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        self.log_dir = os.path.join(self.log_dir, self.model_dir)
        self.dataset_dir = os.path.join(self.dataset_dir, self.dataset_name)


class AdvCrossModalSimple(BaseModel):
    def __init__(self, model_params):
        BaseModel.__init__(self, model_params) 
        self.data_iter = DataIter(self.model_params.batch_size)

        self.visual_feats = tf.placeholder(tf.float32, [None, self.model_params.visual_feat_dim])
        self.word_vecs = tf.placeholder(tf.float32, [None, self.model_params.word_vec_dim])
        self.y = tf.placeholder(tf.int32, [self.model_params.batch_size,10])
        self.y_single = tf.placeholder(tf.int32, [self.model_params.batch_size,1])
        self.l = tf.placeholder(tf.float32, [])
        self.emb_v = self.visual_feature_embed(self.visual_feats)
        self.emb_w = self.label_embed(self.word_vecs)
        #self.corr_loss = tf.sqrt(2 * tf.nn.l2_loss(self.emb_v - self.emb_w))
        #self.corr_loss = tf.reduce_mean(self.corr_loss)
        # dissimilar loss
        emb_v_ = tf.reduce_sum(self.emb_v, axis=1, keep_dims=True)
        emb_w_ = tf.reduce_sum(self.emb_w, axis=1, keep_dims=True)
        distance_map = tf.matmul(emb_v_,tf.ones([1,self.model_params.batch_size])) - tf.matmul(self.emb_v,tf.transpose(self.emb_w))+ \
            tf.matmul(tf.ones([self.model_params.batch_size,1]),tf.transpose(emb_w_))
        mask_initial = tf.to_float(tf.matmul(self.y_single,tf.ones([1,self.model_params.batch_size],dtype=tf.int32)) - \
            tf.matmul(tf.ones([self.model_params.batch_size,1],dtype=tf.int32),tf.transpose(self.y_single)))
        mask = tf.to_float(tf.not_equal(mask_initial, tf.zeros_like(mask_initial)))
        masked_dissimilar_loss = tf.multiply(distance_map,mask)
        self.dissimilar_loss = tf.reduce_mean(tf.maximum(0., 0.1*tf.ones_like(mask)-masked_dissimilar_loss))
        self.similar_loss = tf.sqrt(2 * tf.nn.l2_loss(self.emb_v - self.emb_w))
        self.similar_loss = tf.reduce_mean(self.similar_loss)
        logits_v = self.label_classifier(self.emb_v)
        logits_w = self.label_classifier(self.emb_w, reuse=True)
        self.label_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits_v) + \
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits_w)
        self.label_loss = tf.reduce_mean(self.label_loss)
        self.emb_loss = 50*self.label_loss + self.similar_loss + 0.2*self.dissimilar_loss
        self.emb_v_class = self.domain_classifier(self.emb_v, self.l)
        self.emb_w_class = self.domain_classifier(self.emb_w, self.l, reuse=True)

        all_emb_v = tf.concat([tf.ones([self.model_params.batch_size, 1]),
                                   tf.zeros([self.model_params.batch_size, 1])], 1)
        all_emb_w = tf.concat([tf.zeros([self.model_params.batch_size, 1]),
                                   tf.ones([self.model_params.batch_size, 1])], 1)
        self.domain_class_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.emb_v_class, labels=all_emb_w) + \
            tf.nn.softmax_cross_entropy_with_logits(logits=self.emb_w_class, labels=all_emb_v)
        self.domain_class_loss = tf.reduce_mean(self.domain_class_loss)
        self.t_vars = tf.trainable_variables()
        self.vf_vars = [v for v in self.t_vars if 'vf_' in v.name]
        self.le_vars = [v for v in self.t_vars if 'le_' in v.name]
        self.dc_vars = [v for v in self.t_vars if 'dc_' in v.name]
        self.lc_vars = [v for v in self.t_vars if 'lc_' in v.name]
        

    def visual_feature_embed(self, X, is_training=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = tf.nn.tanh(slim.fully_connected(X, 4096, scope='vf_fc_0'))
            net = tf.nn.tanh(slim.fully_connected(net, 1000, scope='vf_fc_1'))
            net = tf.nn.tanh(slim.fully_connected(net, self.model_params.semantic_emb_dim, scope='vf_fc_2'))
        return net

    def label_embed(self, L, is_training=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = tf.nn.tanh(slim.fully_connected(L, 1000, scope='le_fc_0'))
            net = tf.nn.tanh(slim.fully_connected(net, 300, scope='le_fc_1'))
            net = tf.nn.tanh(slim.fully_connected(net, self.model_params.semantic_emb_dim, scope='le_fc_2'))
        return net 
    def label_classifier(self, X, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = slim.fully_connected(X, 10, scope='lc_fc_0')
        return net         
    def domain_classifier(self, E, l, is_training=True, reuse=False): 
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            E = flip_gradient(E, l)
            net = slim.fully_connected(E, self.model_params.semantic_emb_dim/2, scope='dc_fc_0')
            net = slim.fully_connected(net, self.model_params.semantic_emb_dim/4, scope='dc_fc_1')
            net = slim.fully_connected(net, 2, scope='dc_fc_2')
        return net

    def train(self, sess):
        #self.check_dirs()

        #total_loss = 100*self.emb_loss + 0.05*self.domain_class_loss
        #total_train_op = tf.train.AdamOptimizer(
        #    learning_rate=self.model_params.lr_total,
        #    beta1=0.5).minimize(total_loss)
        emb_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_emb,
            beta1=0.5).minimize(self.emb_loss, var_list=self.le_vars+self.vf_vars)
        domain_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_domain,
            beta1=0.5).minimize(self.domain_class_loss, var_list=self.dc_vars)

        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()
        self.model_params.epoch = 50
        start_time = time.time()
        for epoch in range(self.model_params.epoch):  
        
            p = float(epoch) / self.model_params.epoch
            l = 2. / (1. + np.exp(-10. * p)) - 1
            for batch_feat, batch_vec, batch_labels, batch_labels_single, idx in self.data_iter.train_data():
            
                #sess.run([total_train_op], feed_dict={self.visual_feats: batch_feat, self.word_vecs: batch_vec, self.y: b,self.l: l})
                sess.run([emb_train_op, domain_train_op],
                         feed_dict={
                             self.visual_feats: batch_feat,
                             self.word_vecs: batch_vec,
                             self.y: batch_labels,
                             self.y_single: batch_labels_single[:,np.newaxis],
                             self.l: l})
              
                label_loss_val, similar_loss_val, emb_loss_val, domain_loss_val, dissimilar_loss_val = sess.run([self.label_loss, self.similar_loss, self.emb_loss, self.domain_class_loss, self.dissimilar_loss],
                                                         feed_dict={self.visual_feats: batch_feat,
                                                                    self.word_vecs: batch_vec,
                                                                    self.y: batch_labels,
                                                                    self.y_single: batch_labels_single[:,np.newaxis],
                                                                    self.l: l})
                print('Epoch: [%2d][%4d/%4d] time: %4.4f, emb_loss: %.8f, domain_loss: %.8f, label_loss: %.8f, similar_loss: %.8f, suaimilar_loss: %.8f' %(
                    epoch, idx, self.data_iter.num_train_batch, time.time() - start_time, emb_loss_val, domain_loss_val, label_loss_val, similar_loss_val, dissimilar_loss_val
                ))

    def eval_random_rank(self):
        start = time.time()
        with open('./data/nuswide/test_label_map.pkl', 'rb') as fpkl:
            test_labels = cPickle.load(fpkl)

        k = self.model_params.top_k
        avg_precs = []
        for i in range(len(test_labels)):
            query_label = test_labels[i]

            # distances and sort by distances
            sorted_idx = range(len(test_labels))
            shuffle(sorted_idx)

            # for each k do top-k
            precs = []
            for topk in range(1, k + 1):
                hits = 0
                top_k = sorted_idx[0 : topk]
                if query_label != test_labels[top_k[-1]]:
                    continue
                for ii in top_k:
                    retrieved_label = test_labels[ii]
                    if query_label != retrieved_label:
                        hits += 1
                precs.append(float(hits) / float(topk))
            avg_precs.append(np.sum(precs) / float(k))
        mean_avg_prec = np.mean(avg_precs)
        print('[Eval - random] mAP: %f in %4.4fs' % (mean_avg_prec, (time.time() - start)))

    def eval(self, sess):
        start = time.time()

        test_img_feats_trans = []
        test_txt_vecs_trans = []
        test_labels = []
        for feats, vecs, _, labels, i in self.data_iter.test_data():
            feats_trans = sess.run(self.emb_v, feed_dict={self.visual_feats: feats})
            vecs_trans = sess.run(self.emb_w, feed_dict={self.word_vecs: vecs})
            #print("{0}".format(np.shape(labels)))            
            test_labels += list(labels)
            for ii in range(len(feats)):
                test_img_feats_trans.append(feats_trans[ii])
                test_txt_vecs_trans.append(vecs_trans[ii])
        test_img_feats_trans = np.asarray(test_img_feats_trans)
        test_txt_vecs_trans = np.asarray(test_txt_vecs_trans)
        #print("{0}".format(np.shape(test_labels)))
        print('[Eval] transformed test features in %4.4f' % (time.time() - start))

        avg_precs = []
        all_precs = []
        all_k = [50]
        for k in all_k: 
            for i in range(len(test_txt_vecs_trans)):
                query_label = test_labels[i]

                # distances and sort by distances
                wv = test_txt_vecs_trans[i]
                diffs = test_img_feats_trans - wv
                dists = np.linalg.norm(diffs, axis=1)
                sorted_idx = np.argsort(dists)

                #for each k do top-k
                precs = []
                for topk in range(1, k + 1):
                    hits = 0
                    top_k = sorted_idx[0 : topk]                    
                    if query_label != test_labels[top_k[-1]]:
                        continue
                    for ii in top_k:
                        retrieved_label = test_labels[ii]
                        if retrieved_label == query_label:
                            hits += 1
                    precs.append(float(hits) / float(topk))
                if len(precs) == 0:
                    precs.append(0)
                avg_precs.append(np.average(precs))
            mean_avg_prec = np.mean(avg_precs)
            all_precs.append(mean_avg_prec)
        print('[Eval - txt2img] mAP: %f in %4.4fs' % (all_precs[0], (time.time() - start)))

        avg_precs = []
        all_precs = []
        all_k = [50]
        for k in all_k:        
            for i in range(len(test_img_feats_trans)):
                query_img_feat = test_img_feats_trans[i]
                ground_truth_label = test_labels[i]

                # calculate distance and sort
                diffs = test_txt_vecs_trans - query_img_feat
                dists = np.linalg.norm(diffs, axis=1)
                sorted_idx = np.argsort(dists)

                # for each k in top-k
                precs = []
                for topk in range(1, k + 1):
                    hits = 0
                    top_k = sorted_idx[0 : topk]
                    if ground_truth_label != test_labels[top_k[-1]]:
                        continue
                    for ii in top_k:
                        retrieved_label = test_labels[ii]
                        if ground_truth_label == retrieved_label:
                            hits += 1
                    precs.append(float(hits) / float(topk))
                if len(precs) == 0:
                    precs.append(0)
                avg_precs.append(np.average(precs))
            mean_avg_prec = np.mean(avg_precs)
            all_precs.append(mean_avg_prec)           
        print('[Eval - img2txt] mAP: %f in %4.4fs' % (all_precs[0], (time.time() - start)))
