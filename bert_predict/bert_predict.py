#encoding=utf-8
import tensorflow as tf
import run_classifier_11class
import modeling
import tokenization
import numpy as np

import os
#不让其使用gpu
os.environ['CUDA_VISIBLE_DEVICES']='-1'
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    "bert_config_file", 'checkpointbert/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")
flags.DEFINE_string("vocab_file", 'checkpointbert/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")
flags.DEFINE_string("model_dir11", 'checkpoint11class',
                    "bert训练最新的检测点")
flags.DEFINE_integer(
    "batch_size", 1,"11classfier_predict batch_size")
labels11=['1','2','3','4','5','6','7','8','9','10','11']
tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
class classifier_11class(object):
    def __init__(self):
        tf.reset_default_graph()
        gpu_config=tf.ConfigProto()
        gpu_config.gpu_options.allow_growth=True
        self.sess11=tf.Session(config=gpu_config)
        self.graph11=tf.get_default_graph()
        with self.graph11.as_default():
            print("going to restore 11checkpoint")
            self.input_ids_p=tf.placeholder(tf.int32,[FLAGS.batch_size,FLAGS.max_seq_length],name="input_ids")
            self.input_mask_p=tf.placeholder(tf.int32,[FLAGS.batch_size,FLAGS.max_seq_length],name="input_mask")
            bert_config=modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
            #因为下面是通过self.saver的形式加载模型，因此需要先定义需要操作的变量同时需要先构建图
            #这里传入的label是用作串构建模型时计算loss时用的，而loss值在随后已用不到
            #随后用到的只有预测值，因此这里的label可以随便传一个值，同时需要预测类别的数据本就无类别标签
            (self.total_loss,self.per_example_loss,self.logits,self.pred_ids)=run_classifier_11class.create_model(
                bert_config,False,self.input_ids_p,self.input_mask_p,None,0,len(labels11),False)
            self.saver=tf.train.Saver()
            self.saver.restore(self.sess11,tf.train.latest_checkpoint(FLAGS.model_dir11))
    def predict(self,sentence):
        input_ids, input_mask, segment_ids, label_ids=convert(sentence)
        feed_dict={
            self.input_ids_p:input_ids,
            self.input_mask_p:input_mask
        }
        pred_ids_result=self.sess11.run([self.pred_ids],feed_dict)
        #这里是类别标签的中文名
        bc=['0','1']
        #获得概率值
        probabilities=np.array(pred_ids_result[0][0])
        #选取最大概率的index作为类别标签
        key=np.argmax(probabilities)
        return bc[key]
def convert(line):
    feature=convert_single_example(0,line,labels11,FLAGS.max_seq_length,tokenizer)
    input_ids=np.reshape([feature.input_ids],(FLAGS.batch_size,FLAGS.max_seq_length))
    input_mask=np.reshape([feature.input_mask],(FLAGS.batch_size,FLAGS.max_seq_length))
    segment_ids=np.reshape([feature.segment_ids],(FLAGS.batch_size,FLAGS.max_seq_length))
    label_ids=np.reshape([feature.label_id],(FLAGS.batch_size))
    return input_ids,input_mask,segment_ids,label_ids
class InputFeatures(object):
  """A single set of features of data."""
  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example
#将样本转换成features,这里有改动的地方，对于单文本分类只需传入一个句子
#不需要和源码一样example.text_a，example.text_b
def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i
  #样本向字id的转换
  tokens_a = tokenizer.tokenize(example)
  # Account for [CLS] and [SEP] with "- 2"
  if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  #若输入的是句子对要加入特殊字符
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)
  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  #所有类别标签的第一个类别标签
  label_id = label_map['0']
  #创建InputFeatures的一个实例化对象并返回
  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature