import tensorflow as tf
import numpy as np
import random
import time
import sys
sys.path.append('/search/odin/jdwu/faster_bert/mrc-toolkit/')
from model_seq2seq_contrib import Seq2seq
from train_seq2seq import load_data, get_eval_batch
from train_seq2seq import *
from train_seq2seq import Config
# from model_seq2seq import Seq2seq
from sogou_mrc.libraries import tokenization
vocab_file = '/search/odin/jdwu/chinese_L-12_H-768_A-12/vocab.txt'
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True 

model_path = "checkpoint3/model.ckpt"

if __name__ == "__main__":
	print("(1)load data......")
	# docs_source, docs_target = load_data("/Users/apple/Downloads/yesno_project/dataset/train_data_question_pair")
	#eval_doc_source,eval_doc_target = load_data('/search/odin/jdwu/faster_bert/seq2seq_dataset/dev_data_question_pair')
	eval_doc_source,eval_doc_target = load_data('/search/odin/jdwu/faster_bert/seq2seq_dataset/test_yesno_query.txt')
	#eval_doc_source,eval_doc_target = load_data('/search/odin/jdwu/faster_bert/seq2seq_dataset/test_data_500_question_pair')
	#eval_doc_source,eval_doc_target = load_data('/search/odin/jdwu/faster_bert/seq2seq_dataset/dureader_question_pair')

	print("(2) build model......")
	config = Config()
	config.source_vocab_size = len(tokenizer.vocab)
	config.target_vocab_size = 10000#len(tokenizer.vocab)

	model = Seq2seq(config=config,tokenizer=tokenizer, useTeacherForcing=False, useAttention=True, useBeamSearch=1)

	
	
	print("(3) run model......")
	print_every = 100
	max_target_len = 20
	writer = open('inference_res_0305','a+',encoding="utf-8")
	
	with tf.Session(config=tf_config) as sess:
		saver = tf.train.Saver()
		saver.restore(sess, model_path)
		right_num = 0
		total = 0
		for eval_source_batch, eval_source_lens, eval_target_batch, eval_target_lens in get_eval_batch(eval_doc_source,
																									   eval_doc_target,
																									   batch_size=config.batch_size):
			eval_feed_dict = {
				model.seq_inputs: eval_source_batch,
				model.seq_inputs_length: eval_source_lens,
				#model.seq_targets: eval_target_batch,
				#model.seq_targets_length: eval_target_lens
			}
			predict_batch = sess.run(model.out, eval_feed_dict)
			for i in range(len(eval_source_batch)):
				total += 1
				out = [tok for tok in tokenizer.convert_ids_to_tokens(predict_batch[i]) if
					   tok != "[PAD]" and tok != "[unused2]"]
				gold = [tok for tok in tokenizer.convert_ids_to_tokens(eval_target_batch[i]) if
						tok != "[PAD]" and tok != "[unused2]"]
				if i >=0:
					print(''.join([tok for tok in tokenizer.convert_ids_to_tokens(eval_source_batch[i]) if
								   tok != "[PAD]"]) + "\t" + ''.join(out) + "\t" + ''.join(gold))
					writer.write(''.join([tok for tok in tokenizer.convert_ids_to_tokens(eval_source_batch[i]) if
								   tok != "[PAD]"]) + "\t" + ''.join(out) + "\t" + ''.join(gold)+"\n")
				if ''.join(out) == ''.join(gold):
					right_num += 1
		print(" acc = "+str(float(right_num)/total))
		writer.close()#print(" acc = "+str(float(right_num)/total))
			
