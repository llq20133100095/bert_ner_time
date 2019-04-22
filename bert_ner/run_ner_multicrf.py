#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
@Author:zhoukaiyin
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
from bert import modeling, tokenization, optimization
import tensorflow as tf
import time
import itertools
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import crf_decode
from common import entity
import sys;




reload(sys);
sys.setdefaultencoding("utf8")

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", "./glue/ner",
    "The input datadir.",
)

flags.DEFINE_string(
    "bert_config_file", "./chinese/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", "ner", "The name of the task to train."
)

flags.DEFINE_string("vocab_file", "./chinese/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", "./ner_output/",
    "The output directory where the model checkpoints will be written."
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", "./chinese/bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train", False,
    "Whether to run training."
)
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run predict on the dev set.")
flags.DEFINE_bool("do_demo", True, "Whether to run predict on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 32, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label_type=None, label_abs_rel=None, label_ref=None, label_fre=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label_type = label_type
        self.label_abs_rel = label_abs_rel
        self.label_ref = label_ref
        self.label_fre = label_fre


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_ids_type, label_ids_abs_rel, label_ids_ref, label_ids_fre, input_length):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids_type = label_ids_type
        self.label_ids_abs_rel = label_ids_abs_rel
        self.label_ids_ref = label_ids_ref
        self.label_ids_fre = label_ids_fre
        self.input_length = input_length


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):#xiweichabie
        """Reads a BIO data."""
        with open(input_file) as f:
            lines = []
            words = []
            labels_type = []
            labels_abs_rel = []
            labels_ref = []
            labels_fre = []
            for line in f:
                contends = line.strip()
                if len(contends) == 0:
                    label_type = ' '.join([label_type for label_type in labels_type if len(label_type) > 0])
                    label_abs_rel = ' '.join([label_abs_rel for label_abs_rel in labels_abs_rel if len(label_abs_rel) > 0])
                    label_ref = ' '.join([label_ref for label_ref in labels_ref if len(label_ref) > 0])
                    label_fre = ' '.join([label_fre for label_fre in labels_fre if len(label_fre) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([label_type, label_abs_rel, label_ref, label_fre, w])
                    words = []
                    labels_type = []
                    labels_abs_rel = []
                    labels_ref = []
                    labels_fre = []
                    continue
                segs = contends.split('\t')
                if len(segs) == 5:
                    word = segs[0]
                    label_type = segs[1]
                    label_abs_rel = segs[2]
                    label_ref = segs[3]
                    label_fre = segs[4]

                words.append(word)
                labels_type.append(label_type)
                labels_abs_rel.append(label_abs_rel)
                labels_ref.append(label_ref)
                labels_fre.append(label_fre)
            return lines

    @classmethod
    def _read_demo_data(cls,sentence='自美国2008年推出多轮量化宽松措施以来，香港楼价由2009年起连续五年上升，累计涨幅达到1.1倍'):
        """Reads a BIO data."""
        lines = []
        for line in range(0, 1):
            line = sentence
            contends = line.strip().decode('utf-8')
            l1 = ' '.join([word for word in contends if len(contends) > 0])
            l2 = ' '.join([word for word in contends if len(contends) > 0])
            l3 = ' '.join([word for word in contends if len(contends) > 0])
            l4 = ' '.join([word for word in contends if len(contends) > 0])
            w = ' '.join([word for word in contends if len(contends) > 0])
            lines.append([l1, l2, l3, l4,w])
        return lines

class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_demo_examples(self):
        return self._create_example(
            self._read_demo_data(), "demo"
        )

    def get_demo_examples(self,sentence):
        return self._create_example(
            self._read_demo_data(sentence), "demo"
        )

    def get_labels(self):
        type = ["B-POINT", "I-POINT", "E-POINT", "B-DURATION", "E-DURATION", "I-DURATION", "O", "B-PERIOD", "I-PERIOD",
                "E-PERIOD", "B-FUZZY", "I-FUZZY", "E-FUZZY", "S-FUZZY", "S-POINT", "S-DURATION", "S-FUZZY", "B-PROPE",
                "I-PROPE", "E-PROPE","S-PROPE"]
        abs_rel = ["REL", "ABS", "O"]
        ref = ["T", "O"]
        fre = ["T", "O"]
        return type, abs_rel, ref, fre

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[4])
            label_fre = tokenization.convert_to_unicode(line[3])
            label_ref = tokenization.convert_to_unicode(line[2])
            label_abs_rel = tokenization.convert_to_unicode(line[1])
            label_type = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label_type=label_type, label_abs_rel=label_abs_rel, label_ref=label_ref, label_fre=label_fre))
        return examples


def convert_single_example(ex_index, example, label_list_type, label_list_abs_rel, label_list_ref, label_list_fre,
                           max_seq_length, tokenizer):
    label_map_type = {}
    label_map_absrel = {}
    label_map_ref = {}
    label_map_fre = {}
    for (i, label) in enumerate(label_list_type, 1):
        label_map_type[label] = i
    for (i, label) in enumerate(label_list_abs_rel, 1):
        label_map_absrel[label] = i
    for (i, label) in enumerate(label_list_ref, 1):
        label_map_ref[label] = i
    for (i, label) in enumerate(label_list_fre, 1):
        label_map_fre[label] = i
    textlist = example.text.split(' ')
    labellist_type = example.label_type.split(' ')
    labellist_abs_rel = example.label_abs_rel.split(' ')
    labellist_ref = example.label_ref.split(' ')
    labellist_fre = example.label_fre.split(' ')
    tokens = []
    labels_type = []
    labels_abs_rel = []
    labels_ref = []
    labels_fre = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_type = labellist_type[i]
        label_abs_rel = labellist_abs_rel[i]
        label_ref = labellist_ref[i]
        label_fre = labellist_fre[i]
        for m in range(len(token)):
            if m == 0:
                labels_type.append(label_type)
                labels_abs_rel.append(label_abs_rel)
                labels_ref.append(label_ref)
                labels_fre.append(label_fre)
            else:
                labels_type.append("X")
                labels_abs_rel.append("X")
                labels_ref.append("X")
                labels_fre.append("X")
    # tokens = tokenizer.tokenize(example.text)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids_type = []
    label_ids_abs_rel = []
    label_ids_ref = []
    label_ids_fre = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids_type.append(0)
    label_ids_abs_rel.append(0)
    label_ids_ref.append(0)
    label_ids_fre.append(0)
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        # label_ids_type.append(label_map_type[labels_type[i]])
        # label_ids_abs_rel.append(label_map_absrel[labels_abs_rel[i]])
        # label_ids_ref.append(label_map_ref[labels_ref[i]])
        # label_ids_fre.append(label_map_fre[labels_fre[i]])

        label_ids_type.append(0)
        label_ids_abs_rel.append(0)
        label_ids_ref.append(0)
        label_ids_fre.append(0)
    ntokens.append("[SEP]")
    segment_ids.append(0)
    label_ids_type.append(0)
    label_ids_abs_rel.append(0)
    label_ids_ref.append(0)
    label_ids_fre.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids_type.append(0)
        label_ids_abs_rel.append(0)
        label_ids_ref.append(0)
        label_ids_fre.append(0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids_type) == max_seq_length
    assert len(label_ids_abs_rel) == max_seq_length
    assert len(label_ids_ref) == max_seq_length
    assert len(label_ids_fre) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids_type: %s" % " ".join([str(x) for x in label_ids_type]))
        tf.logging.info("label_ids_abs_rel: %s" % " ".join([str(x) for x in label_ids_abs_rel]))
        tf.logging.info("label_ids_ref: %s" % " ".join([str(x) for x in label_ids_ref]))
        tf.logging.info("label_ids_fre: %s" % " ".join([str(x) for x in label_ids_fre]))
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids_type=label_ids_type,
        label_ids_abs_rel=label_ids_abs_rel,
        label_ids_ref=label_ids_ref,
        label_ids_fre=label_ids_fre,
        input_length=len(ntokens)
    )
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list_type, label_list_abs_rel, label_list_ref, label_list_fre, max_seq_length, tokenizer, output_file
):
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list_type, label_list_abs_rel, label_list_ref, label_list_fre, max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids_type"] = create_int_feature(feature.label_ids_type)
        features["label_ids_abs_rel"] = create_int_feature(feature.label_ids_abs_rel)
        features["label_ids_ref"] = create_int_feature(feature.label_ids_ref)
        features["label_ids_fre"] = create_int_feature(feature.label_ids_fre)
        features["input_length"] = create_feature(feature.input_length)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids_type": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids_abs_rel": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids_ref": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids_fre": tf.FixedLenFeature([seq_length], tf.int64),
        "input_length": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d
    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels_type, labels_abs_rel, labels_ref, labels_fre, num_labels_type,
                 num_labels_abs_rel, num_labels_ref, num_labels_fre, use_one_hot_embeddings,input_length):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value

    output_weights_type = tf.get_variable(
        "output_weights_type", [num_labels_type, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_weights_abs_rel = tf.get_variable(
        "output_weights_abs_rel", [num_labels_abs_rel, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_weights_ref = tf.get_variable(
        "output_weights_ref", [num_labels_ref, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_weights_fre = tf.get_variable(
        "output_weights_fre", [num_labels_fre, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )

    output_bias_type = tf.get_variable(
        "output_bias_type", [num_labels_type], initializer=tf.zeros_initializer()
    )
    output_bias_abs_rel = tf.get_variable(
        "output_bias_abs_rel", [num_labels_abs_rel], initializer=tf.zeros_initializer()
    )
    output_bias_ref = tf.get_variable(
        "output_bias_ref", [num_labels_ref], initializer=tf.zeros_initializer()
    )
    output_bias_fre = tf.get_variable(
        "output_bias_fre", [num_labels_fre], initializer=tf.zeros_initializer()
    )

    if is_training:
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    output_layer = tf.reshape(output_layer, [-1, hidden_size])
    with tf.variable_scope("loss_type"):
        logits_type = tf.matmul(output_layer, output_weights_type, transpose_b=True)
        logits_type = tf.nn.bias_add(logits_type, output_bias_type)
        logits_type = tf.reshape(logits_type, [-1, FLAGS.max_seq_length, 21])

        log_likelihood_type, transition_params_type = crf_log_likelihood(inputs=logits_type,tag_indices=labels_type, sequence_lengths=input_length)
        loss_type = -tf.reduce_mean(log_likelihood_type)
        pres_id_type, _ = crf_decode(logits_type, transition_params_type, input_length)

    with tf.variable_scope("loss_abs_rel"):
        logits_abs_rel = tf.matmul(output_layer, output_weights_abs_rel, transpose_b=True)
        logits_abs_rel = tf.nn.bias_add(logits_abs_rel, output_bias_abs_rel)
        logits_abs_rel = tf.reshape(logits_abs_rel, [-1, FLAGS.max_seq_length, 3])

        log_likelihood_abs_rel, transition_params_abs_rel = crf_log_likelihood(inputs=logits_abs_rel, tag_indices=labels_abs_rel,
                                                                   sequence_lengths=input_length)
        loss_abs_rel = -tf.reduce_mean(log_likelihood_abs_rel)
        pres_id_abs_rel, _ = crf_decode(logits_abs_rel, transition_params_abs_rel, input_length)

    with tf.variable_scope("loss_ref"):
        logits_ref = tf.matmul(output_layer, output_weights_ref, transpose_b=True)
        logits_ref = tf.nn.bias_add(logits_ref, output_bias_ref)
        logits_ref = tf.reshape(logits_ref, [-1, FLAGS.max_seq_length, 2])

        log_likelihood_ref, transition_params_ref = crf_log_likelihood(inputs=logits_ref, tag_indices=labels_ref,
                                                                   sequence_lengths=input_length)
        loss_ref = -tf.reduce_mean(log_likelihood_ref)
        pres_id_ref, _ = crf_decode(logits_ref, transition_params_ref, input_length)

    with tf.variable_scope("loss_fre"):
        logits_fre = tf.matmul(output_layer, output_weights_fre, transpose_b=True)
        logits_fre = tf.nn.bias_add(logits_fre, output_bias_fre)
        logits_fre = tf.reshape(logits_fre, [-1, FLAGS.max_seq_length, 2])

        log_likelihood_fre, transition_params_fre = crf_log_likelihood(inputs=logits_fre, tag_indices=labels_fre,
                                                                   sequence_lengths=input_length)
        loss_fre = -tf.reduce_mean(log_likelihood_fre)
        pres_id_fre, _ = crf_decode(logits_fre, transition_params_fre, input_length)

    loss = loss_type + loss_abs_rel + loss_ref + loss_fre
    return (loss,pres_id_type, pres_id_abs_rel, pres_id_ref, pres_id_fre)



def model_fn_builder(bert_config, num_labels_type, num_labels_abs_rel, num_labels_ref, num_labels_fre, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids_type = features["label_ids_type"]
        label_ids_abs_rel = features["label_ids_abs_rel"]
        label_ids_ref = features["label_ids_ref"]
        label_ids_fre = features["label_ids_fre"]
        input_length = features["input_length"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (loss, pres_id_type, pres_id_abs_rel, pres_id_ref, pres_id_fre) = create_model(bert_config, is_training,
            input_ids, input_mask, segment_ids, label_ids_type,label_ids_abs_rel,label_ids_ref,label_ids_fre,
            num_labels_type, num_labels_abs_rel, num_labels_ref, num_labels_fre, use_one_hot_embeddings, input_length)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            # tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)


        tf.logging.info("**** Trainable Variables ****")

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        else:
            predict_output_type = {'values_type': pres_id_type, 'values_abs_rel': pres_id_abs_rel,
                                   'values_ref': pres_id_ref, 'values_fre': pres_id_fre}
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predict_output_type, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "conll": NerProcessor,
        "ner": NerProcessor
    }
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict and not FLAGS.do_demo:
        raise ValueError("At least one of `do_train` or `do_eval` or 'do_predict' or 'do_demo'must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list_type, label_list_abs_rel, label_list_ref, label_list_fre = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels_type=len(label_list_type),
        num_labels_abs_rel=len(label_list_abs_rel),
        num_labels_ref=len(label_list_ref),
        num_labels_fre=len(label_list_fre),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=FLAGS.eval_batch_size,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        filed_based_convert_examples_to_features(
            train_examples, label_list_type, label_list_abs_rel, label_list_ref, label_list_fre, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_predict:
        begin_time = time.time()
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        filed_based_convert_examples_to_features(
            eval_examples, label_list_type, label_list_abs_rel, label_list_ref, label_list_fre, FLAGS.max_seq_length,
            tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_steps = None
        if FLAGS.use_tpu:
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
        results = estimator.predict(input_fn=eval_input_fn)
        real_labels_type = []
        real_labels_abs_rel = []
        real_labels_ref = []
        real_labels_fre = []
        text_list = []
        for example in eval_examples:
            label_type = example.label_type.split(' ')
            label_abs_rel = example.label_abs_rel.split(' ')
            label_ref = example.label_ref.split(' ')
            label_fre = example.label_fre.split(' ')
            text = example.text.split(' ')
            text_list.append(text)
            real_labels_type.append(label_type)
            real_labels_abs_rel.append(label_abs_rel)
            real_labels_ref.append(label_ref)
            real_labels_fre.append(label_fre)
        get_eval(results, real_labels_type, real_labels_abs_rel, real_labels_ref, real_labels_fre, text_list,
                 label_list_type, label_list_abs_rel, label_list_ref, label_list_fre)

    if FLAGS.do_demo:
        begin_time1 = time.time()
        print(begin_time1)
        eval_examples = processor.get_demo_examples("")
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        filed_based_convert_examples_to_features(
            eval_examples, label_list_type, label_list_abs_rel, label_list_ref, label_list_fre, FLAGS.max_seq_length,
            tokenizer, eval_file)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        results = estimator.predict(input_fn=eval_input_fn)
        text_list = []
        for example in eval_examples:
            text = example.text.split(' ')
            text_list.append(text)
        get_eval_demo(results, text_list,label_list_type, label_list_abs_rel, label_list_ref, label_list_fre)
        end_time = time.time()
        print(str(end_time - begin_time1) + " s")

def get_eval_demo(results, text_list,label_list_type, label_list_abs_rel, label_list_ref, label_list_fre):
    label_map_type = {}
    for (i, label) in enumerate(label_list_type):
        label_map_type[i + 1] = label

    label_map_abs_rel = {}
    for (i, label) in enumerate(label_list_abs_rel):
        label_map_abs_rel[i + 1] = label

    label_map_ref = {}
    for (i, label) in enumerate(label_list_ref):
        label_map_ref[i + 1] = label

    label_map_fre = {}
    for (i, label) in enumerate(label_list_fre):
        label_map_fre[i + 1] = label

    predictions = list(itertools.islice(results, len(text_list)))

    pred_labels_type = []
    pred_labels_abs_rel = []
    pred_labels_ref = []
    pred_labels_fre = []
    for i in range(len(predictions)):
        pred_type = predictions[i]['values_type'][1:len(text_list[i]) + 1]
        pred_abs_rel = predictions[i]['values_abs_rel'][1:len(text_list[i]) + 1]
        pred_ref = predictions[i]['values_ref'][1:len(text_list[i]) + 1]
        pred_fre = predictions[i]['values_fre'][1:len(text_list[i]) + 1]

        pred_ = []
        for l in pred_type:
            if l == 0:
                pred_.append(label_map_type[7])
            else:
                pred_.append(label_map_type[l])
        pred_labels_type.append(pred_)

        pred_ = []
        for l in pred_abs_rel:
            if l == 0:
                pred_.append(label_map_abs_rel[3])
            else:
                pred_.append(label_map_abs_rel[l])
        pred_labels_abs_rel.append(pred_)

        pred_ = []
        for l in pred_ref:
            if l == 0:
                pred_.append(label_map_ref[2])
            else:
                pred_.append(label_map_ref[l])
        pred_labels_ref.append(pred_)

        pred_ = []
        for l in pred_fre:
            if l == 0:
                pred_.append(label_map_fre[2])
            else:
                pred_.append(label_map_fre[l])
        pred_labels_fre.append(pred_)

    pred_entities = get_entity(pred_labels_type, pred_labels_abs_rel, pred_labels_ref, pred_labels_fre, text_list)

    f1 = open('./ner_output/demo_reult.txt', 'w')
    for i, sen in enumerate(zip(text_list)):
        sen = ''.join(sen[0])
        list_en = pred_entities[i]

        f1.write(u"%d: " % (i + 1))
        f1.write(u"原句子：%s \n " % (sen.decode('utf-8')))
        f1.write(u"实体信息 ：\n")
        if len(list_en) > 0:
            for entity in list_en:
                f1.write(u"实体:" + entity.entity.decode('utf-8'))
                f1.write(u"  类型:" + entity.type)
                if entity.abs_rel != None:
                    f1.write(u"  相对绝对:" + entity.abs_rel)
                f1.write(u"  指代: " + str(entity.is_refer))
                f1.write(u"  频率: " + str(entity.is_freq))
                f1.write("\n")
        f1.write("\n")

def get_eval(pred_results, real_labels_type, real_labels_abs_rel, real_labels_ref, real_labels_fre, text_list,
             label_list_type, label_list_abs_rel, label_list_ref, label_list_fre):
    label_map_type = {}
    for (i, label) in enumerate(label_list_type):
        label_map_type[i+1] = label

    label_map_abs_rel = {}
    for (i, label) in enumerate(label_list_abs_rel):
        label_map_abs_rel[i + 1] = label

    label_map_ref = {}
    for (i, label) in enumerate(label_list_ref):
        label_map_ref[i + 1] = label

    label_map_fre = {}
    for (i, label) in enumerate(label_list_fre):
        label_map_fre[i + 1] = label

    predictions = list(itertools.islice(pred_results, len(real_labels_type)))

    pred_labels_type = []
    pred_labels_abs_rel= []
    pred_labels_ref = []
    pred_labels_fre = []
    for i in range(len(predictions)):
        pred_type = predictions[i]['values_type'][1:len(real_labels_type[i])+1]
        pred_abs_rel = predictions[i]['values_abs_rel'][1:len(real_labels_type[i]) + 1]
        pred_ref = predictions[i]['values_ref'][1:len(real_labels_type[i]) + 1]
        pred_fre = predictions[i]['values_fre'][1:len(real_labels_type[i]) + 1]

        pred_ = []
        for l in pred_type:
            if l == 0:
                pred_.append(label_map_type[7])
            else:
                pred_.append(label_map_type[l])
        pred_labels_type.append(pred_)

        pred_ = []
        for l in pred_abs_rel:
            if l == 0:
                pred_.append(label_map_abs_rel[3])
            else:
                pred_.append(label_map_abs_rel[l])
        pred_labels_abs_rel.append(pred_)

        pred_ = []
        for l in pred_ref:
            if l == 0:
                pred_.append(label_map_ref[2])
            else:
                pred_.append(label_map_ref[l])
        pred_labels_ref.append(pred_)

        pred_ = []
        for l in pred_fre:
            if l == 0:
                pred_.append(label_map_fre[2])
            else:
                pred_.append(label_map_fre[l])
        pred_labels_fre.append(pred_)

    pred_entities = get_entity(pred_labels_type, pred_labels_abs_rel, pred_labels_ref, pred_labels_fre, text_list)
    real_entities = get_entity(real_labels_type, real_labels_abs_rel, real_labels_ref, real_labels_fre, text_list)

    # sum_model = 0
    # sum_input = 0
    # correct = 0
    # for i, sen in enumerate(zip(text_list)):
    #     pred_entity = pred_entities[i]
    #     input_entity = real_entities[i]
    #     # sum_model = sum_model + len(pred_entity)
    #     # sum_input = sum_input + len(input_entity)
    #     for j in range(len(pred_entity)):
    #         tt = -1
    #         for k in range(len(input_entity)):
    #             if pred_entity[j].entity == input_entity[k].entity:
    #                 sum_model = sum_model +1
    #                 if pred_entity[j].entity == input_entity[k].entity and pred_entity[j].abs_rel == input_entity[k].abs_rel\
    #                         and pred_entity[j].is_refer == input_entity[k].is_refer and pred_entity[j].is_freq == input_entity[k].is_freq:
    #                     correct = correct + 1
    #                     tt = k
    #                     break
    #         if tt != -1:
    #             del input_entity[tt]
    #
    # print(correct)
    # print(sum_model)
    # # print(sum_input)
    # print("entity precision is: " + str(correct * 1.0 / sum_model))
    # # print("entity recall_scorell is: " + str(correct * 1.0 / sum_input))

    f1 = open('./ner_output/duration15000-20000.txt', 'w')
    for i, sen in enumerate(zip(text_list)):
        sen = ''.join(sen[0])
        list_en = pred_entities[i]
        input_list_en = real_entities[i]

        if len(input_list_en) == len(list_en):
            flag = True

            for j in range(len(list_en)):
                tt = False
                for k in range(len(input_list_en)):
                    if list_en[j].entity == input_list_en[k].entity:
                        tt = True
                if not tt:
                    flag = False
            if not flag:
                f1.write(u"%d: " % (i + 1))
                f1.write(u"原句子：%s \n " % (sen.decode('utf-8')))
                f1.write(u"期望实体信息 ：\n")
                if len(input_list_en) > 0:
                    for entity in input_list_en:
                        f1.write(u"实体:" + entity.entity.decode('utf-8'))
                        f1.write(u"  类型:" + entity.type)
                        if entity.abs_rel != None:
                            f1.write(u"  相对绝对:" + entity.abs_rel)

                        f1.write(u"  指代: " + str(entity.is_refer))
                        f1.write(u"  频率: " + str(entity.is_freq))
                        f1.write("\n")
                if len(list_en) > 0:
                    f1.write(u"实际实体信息 ：\n")
                    for entity in list_en:
                        f1.write(u"实体:" + entity.entity.decode('utf-8'))
                        f1.write(u"  类型:" + entity.type)
                        if entity.abs_rel != None:
                            f1.write(u"  相对绝对:" + entity.abs_rel)
                        f1.write(u"  指代: " + str(entity.is_refer))
                        f1.write(u"  频率: " + str(entity.is_freq))
                        f1.write("\n")
                f1.write("\n")

        else:
            f1.write(u"%d: " %(i+1))
            f1.write(u"原句子：%s \n " %(sen.decode('utf-8')))
            f1.write(u"期望实体信息 ：\n")
            if len(input_list_en) > 0:
                for entity in input_list_en:
                    f1.write(u"实体:" + entity.entity.decode('utf-8'))
                    f1.write(u"类型:" + str(entity.type))
                    if entity.abs_rel != None:
                        f1.write(u"  相对绝对:" + entity.abs_rel)
                    f1.write(u"  指代: " + str(entity.is_refer))
                    f1.write(u"  频率: " + str(entity.is_freq))
                    f1.write("\n")
            if len(list_en) > 0:
                f1.write(u"实际实体信息 ：\n")
                for entity in list_en:
                    f1.write(u"实体:" + entity.entity.decode('utf-8'))
                    f1.write(u"  类型:" + entity.type)
                    if entity.abs_rel != None:
                        f1.write(u"  相对绝对:" + entity.abs_rel)
                    f1.write(u"  指代: " + str(entity.is_refer))
                    f1.write(u"  频率: " + str(entity.is_freq))
                    f1.write("\n")
            f1.write("\n")

def get_entity(type_tag, fre_tag, ref_tag, relab_tag, char_seq):
    entities = get_type_entity(char_seq, type_tag)
    entities = get_Dimension_info(entities, fre_tag, ref_tag, relab_tag)
    return entities

def get_Dimension_info(entities, fre_tags, ref_tags, relab_tags):
    for entity_list, fre_tag_list, ref_tag_list, relab_tag_list in zip(entities, fre_tags, ref_tags, relab_tags):
        for entity in entity_list:
            start = entity.start
            end = entity.end
            fre = list(filter(lambda x: True if fre_tag_list[x] == u'T' else False, range(start,end)))
            if len(fre) == end - start:
                entity.is_freq = True
            ref = list(filter(lambda x: True if ref_tag_list[x] == u'T' else False, range(start, end)))
            if len(ref) == end - start:
                entity.is_refer = True
            abs = list(filter(lambda x: True if relab_tag_list[x] == u'ABS' else False, range(start, end)))
            if len(abs) == end - start:
                entity.abs_rel = 'absolute'
            rel = list(filter(lambda x: True if relab_tag_list[x] == u'REL' else False, range(start, end)))
            if len(rel) == end - start:
                entity.abs_rel = 'relative'
    return entities

type_dict = {'POINT':'time_point', 'PERIOD':'time_period', 'DURATION':'time_duration', 'FUZZY':'time_fuzzy', 'PROPE':'time_prope'}
def get_type_entity(char_seqs, type_tags):
    list_entities = []
    for i, (type_tag, char_seq) in enumerate(zip(type_tags, char_seqs)):
        entity_seq, entities = [], []
        instance = entity.entity()
        for i, (char, tag) in enumerate(zip(char_seq, type_tag)):
            if "B-" in tag:
                instance = entity.entity()
                entity_seq = []
            if 'POINT' in tag:
                entity_tagger(instance, char, entity_seq, i, tag, 'POINT')
            elif 'PERIOD' in tag:
                entity_tagger(instance, char, entity_seq, i, tag, 'PERIOD')
            elif 'DURATION' in tag:
                entity_tagger(instance, char, entity_seq, i, tag, 'DURATION')
            elif 'FUZZY' in tag:
                entity_tagger(instance, char, entity_seq, i, tag, 'FUZZY')
            elif 'PROPE' in tag:
                entity_tagger(instance, char, entity_seq, i, tag, 'PROPE')
            if validate_entity(instance):
                entities.append(instance)
                instance = entity.entity()
                entity_seq = []
        list_entities.append(entities)
    return list_entities

def entity_tagger(entity, char, entity_seq, i, tag, type):
    if tag == 'B-'+type:
        entity.start = i
        entity_seq.append(char)
    # elif entity.type == type_dict[type]:
    elif tag == 'I-'+type:
        entity_seq.append(char)
    elif tag == 'S-'+type:
        entity.start = i
        entity.end = i + 1
        entity.entity = char
    elif tag == 'E-'+type:
        entity.type = type_dict[type]
        entity.end = i + 1
        entity_seq.append(char)
        entity.entity = ''.join(entity_seq)

def validate_entity(entity):
    if entity.entity and entity.end >= entity.start >=0:
        if len(entity.entity) == entity.end - entity.start:
            return True
    return False

def tag_recognize(sentence):

    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "conll": NerProcessor,
        "ner": NerProcessor
    }
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict and not FLAGS.do_demo:
        raise ValueError("At least one of `do_train` or `do_eval` or 'do_predict' or 'do_demo'must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list_type, label_list_abs_rel, label_list_ref, label_list_fre = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels_type=len(label_list_type),
        num_labels_abs_rel=len(label_list_abs_rel),
        num_labels_ref=len(label_list_ref),
        num_labels_fre=len(label_list_fre),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=FLAGS.eval_batch_size,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size)

    begin_time1 = time.time()
    print(begin_time1)
    eval_examples = processor.get_demo_examples(sentence)
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    filed_based_convert_examples_to_features(
        eval_examples, label_list_type, label_list_abs_rel, label_list_ref, label_list_fre, FLAGS.max_seq_length,
        tokenizer, eval_file)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    results = estimator.predict(input_fn=eval_input_fn)
    text_list = []
    for example in eval_examples:
        text = example.text.split(' ')
        text_list.append(text)
    # get_eval_demo(results, text_list, label_list_type, label_list_abs_rel, label_list_ref, label_list_fre)
    end_time = time.time()
    print(str(end_time - begin_time1) + " s")
    return results, text_list, label_list_type, label_list_abs_rel, label_list_ref, label_list_fre


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()


