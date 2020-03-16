import tensorflow as tf
import sys
sys.path.append('faster_bert/mrc-toolkit/')
import numpy as np
import random
import time
from model_seq2seq_contrib import Seq2seq
import jieba
import sys
from sogou_mrc.libraries import tokenization
vocab_file = 'chinese_L-12_H-768_A-12/vocab.txt'
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

import sys
# from model_seq2seq import Seq2seq

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True 


PAD_ID = tokenizer.convert_tokens_to_ids(["[PAD]"])[0]
GO_ID = tokenizer.convert_tokens_to_ids(["[unused1]"])[0]
EOS_ID = tokenizer.convert_tokens_to_ids(["[unused2]"])[0]
class Config(object):
	embedding_dim = 100
	hidden_dim = 256
	batch_size = 32
	learning_rate = 0.001
	source_vocab_size = None
	target_vocab_size = None


def load_data(path='train_data_question_pair'):
	docs_source = []
	docs_target = []
	with open(path,'r',encoding="utf-8") as lines:
		for idx,line in enumerate(lines):
			#if idx>=10:break

			data = line.strip().split('\t')
			if len(data)<3:continue #print(data)#if idx>=10:break
			query = data[0]+'[SEP]'+data[1] if 'dureader' not in path else  data[0]+'[SEP]'+data[2] 
			doc_source  = tokenizer.tokenize(query)
			doc_target = tokenizer.tokenize(data[2]) if 'dureader' not in path else tokenizer.tokenize(data[1])
			docs_source.append(doc_source)
			docs_target.append(doc_target)
	
	return docs_source, docs_target




def get_batch(docs_source, docs_target, batch_size,is_training=True):
	ps = []
	if is_training:
		while len(ps) < batch_size:
			ps.append(random.randint(0, len(docs_source)-1))
	source_batch = []
	target_batch = []
	
	source_lens = [len(docs_source[p]) for p in ps]
	target_lens = [len(docs_target[p])+1 for p in ps]
	
	max_source_len = max(source_lens)
	max_target_len = max(target_lens)
	import sys
	for p in ps:

		source_seq = tokenizer.convert_tokens_to_ids( docs_source[p] )+ [PAD_ID]*(max_source_len-len(docs_source[p]))
		target_seq = tokenizer.convert_tokens_to_ids(docs_target[p])+ [EOS_ID] + [PAD_ID]*(max_target_len-1-len(docs_target[p]))
		target_seq = [tok_id if tok_id<10000 else 100 for tok_id in target_seq] #source_batch.append(source_seq)
		source_batch.append(source_seq)
		target_batch.append(target_seq)
	
	return source_batch, source_lens, target_batch, target_lens

def get_eval_batch(docs_source,docs_target,batch_size=16):
	sindex = 0
	eindex = batch_size
	while eindex < len(docs_target):
		target_batch = docs_target[sindex:eindex]
		source_batch = docs_source[sindex:eindex]
		source_lens = [len(source) for source in source_batch]
		target_lens = [len(target)+1 for target in target_batch]
		max_source_len = max(source_lens)
		max_target_len = max(target_lens)
		new_source_batch = []
		new_target_batch = []
		for idx in range(len(target_batch)):
			source_seq = tokenizer.convert_tokens_to_ids(source_batch[idx])+[PAD_ID]*(max_source_len-len(source_batch[idx]))
			target_seq = tokenizer.convert_tokens_to_ids(target_batch[idx])+[EOS_ID] + [PAD_ID]*(max_target_len-1-len(target_batch[idx]))

			new_source_batch.append(source_seq)
			new_target_batch.append([tok_id if tok_id<10000 else 100 for tok_id in target_seq])
		temp = eindex
		eindex = eindex + batch_size
		sindex = temp
		yield (new_source_batch,source_lens,new_target_batch,target_lens)




# for batch in  get_eval_batch([i for i in range(9)],[i for i in range(9)],4):
# 	print(batch)
# sys.exit(1)
	
	
if __name__ == "__main__":

	print("(1)load data......")
	docs_source, docs_target = load_data('seq2seq_dataset/train_data_question_pair')
	eval_doc_source,eval_doc_target = load_data('seq2seq_dataset/dev_data_question_pair')

	print("(2) build model......")
	config = Config()
	config.source_vocab_size = len(tokenizer.vocab)
	config.target_vocab_size = 10000#len(tokenizer.vocab)


	model = Seq2seq(config=config,tokenizer=tokenizer, useTeacherForcing=True, useAttention=True, useBeamSearch=1)

	print("(3) run model......")
	print_every = 200
	epoches = 3
	train_data_size = len(docs_target)
	batch_size = 32
	batches = train_data_size//batch_size

	acc = 0.0
	with tf.Session(config=tf_config) as sess:
		tf.summary.FileWriter('graph', sess.graph)
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())
		
		losses = []
		total_loss = 0
		for epoch in range(epoches):
			for batch in range(batches):
				source_batch, source_lens, target_batch, target_lens = get_batch(docs_source,docs_target,config.batch_size)

				feed_dict = {
					model.seq_inputs: source_batch,
					model.seq_inputs_length: source_lens,
					model.seq_targets: target_batch,
					model.seq_targets_length: target_lens
				}

				loss, _ = sess.run([model.loss, model.train_op], feed_dict)
				total_loss += loss

				if batch % print_every == 0 and batch > 0:
					print_loss = total_loss if batch == 0 else total_loss / print_every
					losses.append(print_loss)
					total_loss = 0
					print("-----------------------------")
					print("batch:",batch,"/",batches)
					print("time:",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
					print("loss:",print_loss)
					right_num = 0
					total = 0
					for eval_source_batch,eval_source_lens,eval_target_batch,eval_target_lens in get_eval_batch(eval_doc_source,eval_doc_target,batch_size=config.batch_size):
						eval_feed_dict = {
							model.seq_inputs: eval_source_batch,
							model.seq_inputs_length: eval_source_lens,
							model.seq_targets: eval_target_batch,
							model.seq_targets_length: eval_target_lens
						}
						predict_batch = sess.run(model.out,eval_feed_dict)
						for i in range(len(eval_source_batch)):

							total+=1
							out = [tok for tok in tokenizer.convert_ids_to_tokens(predict_batch[i]) if tok != "[PAD]" and tok!="[unused2]"]
							gold = [tok for tok in tokenizer.convert_ids_to_tokens(eval_target_batch[i]) if tok != "[PAD]" and tok!="[unused2]"]
							if i==2:
								print(''.join([tok for tok in tokenizer.convert_ids_to_tokens(eval_source_batch[i]) if tok != "[PAD]"])+"\t"+''.join(out)+"\t"+''.join(gold))
							if ''.join(out)==''.join(gold):
								right_num+=1

				# for i in range(3):
				# 	print("in:", [tok for tok in tokenizer.convert_ids_to_tokens(source_batch[i]) if tok != "[PAD]"])
				# 	print("out:",[tok for tok in tokenizer.convert_ids_to_tokens(predict_batch[i]) if tok != "[PAD]"])
				# 	print("tar:",[tok for tok in tokenizer.convert_ids_to_tokens(target_batch[i]) if tok != "[PAD]"])
					print("predict gold answer acc {:4f}".format(float(right_num)/total))
					#acc = max(acc,float(right_num)/total)
					t_acc = float(right_num)/total
					if t_acc>acc:#sys.exit(1)
						acc = t_acc#for i in range(len(eval_source_batch)):
						saver.save(sess,"checkpoint3/model.ckpt")#acc = t_acc#for i in range(len(eval_source_batch)):


				
					# print("samples:\n")
					# predict_batch = sess.run(model.out, feed_dict)
					# for i in range(3):
					# 	print("in:", [tok for tok in tokenizer.convert_ids_to_tokens(source_batch[i]) if tok != "[PAD]"])
					# 	print("out:",[tok for tok in tokenizer.convert_ids_to_tokens(predict_batch[i]) if tok != "[PAD]"])
					# 	print("tar:",[tok for tok in tokenizer.convert_ids_to_tokens(target_batch[i]) if tok != "[PAD]"])
					# 	print("")

		#print(losses)
		print(" max acc = " +str(acc))#print(losses)
		#print(saver.save(sess, "checkpoint2/model.ckpt"))
		
	


