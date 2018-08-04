import numpy as np
import re
import jieba
import pandas as pd
import random
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors, doc2vec

# 读取数据
t_q_s = pd.read_csv('data/tiku_question_sx.csv')
k_h = pd.read_csv('data/knowledge_hierarchy.csv')
q_k_h_s = pd.read_csv('data/question_knowledge_hierarchy_sx.csv')

# 读取中文停止词库(https://github.com/chdd/weibo/blob/master/stopwords/中文停用词库.txt)
stopwords = '|'.join([line.strip() for line in open('stopwords.txt', 'r')])

# 去除问题中所有非中文字符和停用词（stopwords）,并进行分词
def clean(question):
	question = re.sub(r'[^\u4e00-\u9fa5]', '', question)
	question = re.sub(stopwords, '', question)
	return ' '.join(jieba.cut(question))

##################################################################################################
##                                            Task1                                             ##
##                   Computing similarity using bag-of-words representation                     ##
##################################################################################################

# 读取题目并清洗
questions_uncleaned = [str(line) for line in t_q_s['content'].tolist()]
questions = [clean(q) for q in questions_uncleaned]

# 提取TF和TF-IDF特征
vectorizer = CountVectorizer()
tf = vectorizer.fit_transform(questions)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(tf)

# 随机抽取5道题目，对其中每一道找出题库中与其最相似的3道题目
for i in range(5):
	id = random.randint(0, len(questions)-1)
	sample = questions_uncleaned[id]
	print('Sample question: ' + sample)
	for feat in ['tf', 'tfidf']:
		mat = globals()[feat]
		print('Using ' + feat + ' as feature...')
		cos_mat = cosine_similarity(mat[id], mat).reshape((-1,))
		id_list = np.argpartition(cos_mat, -4)[-4:-1][::1]
		for id in id_list:
			print(questions_uncleaned[id])
		print('\n')

##################################################################################################
##                                           Task2                                              ##
##                 Sentence classification with SVM and Logistic Regression                     ##
##################################################################################################

# 筛选出一级知识点为'三角函数与解三角形'的题目
tri_id = k_h[k_h['name'] == '三角函数与解三角形']['id'].values[0]
tri_family = np.append(k_h[k_h['root_id'] == tri_id]['id'].values, tri_id)
tri_qid = q_k_h_s[q_k_h_s['kh_id'].isin(tri_family)]['question_id'].values
tri_q_uncleaned = t_q_s[t_q_s['que_id'].isin(tri_qid)]['content'].tolist()
tri_q = [clean(q) for q in tri_q_uncleaned]

# 筛选出一级知识点为'函数与导数'的题目
func_id = k_h[k_h['name'] == '函数与导数']['id'].values[0]
func_family = np.append(k_h[k_h['root_id'] == func_id]['id'].values, func_id)
func_qid = q_k_h_s[q_k_h_s['kh_id'].isin(func_family)]['question_id'].values
func_q_uncleaned = t_q_s[t_q_s['que_id'].isin(func_qid)]['content'].tolist()
func_q = [clean(q) for q in func_q_uncleaned]

# 建立训练集与测试集
X = np.append(tri_q, func_q)
Y = [0]*len(tri_q)+[1]*len(func_q)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)

# 提取TF-IDF特征
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# 训练及测试模型
# clf = LinearSVC()
clf = LogisticRegression()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

# 评估模型表现
print('Precision: ', precision_score(Y_test, Y_pred))
print('Recall: ', recall_score(Y_test, Y_pred))
print('F1 score: ', f1_score(Y_test, Y_pred))
print('Accuracy: ', accuracy_score(Y_test, Y_pred))
print('ROC_AUC score: ', roc_auc_score(Y_test, clf.decision_function(X_test)))

##################################################################################################
##                                           Task3                                              ##
##                 Computing semantic similarity using pre-trained word embeddings              ##
##################################################################################################

# 提取知识点并进行清理
know_uncleaned = [str(line) for line in k_h['name'].tolist()]
know = [clean(k) for k in know_uncleaned]

# 建立词汇表，过滤词向量
V = set().union(*(k.split() for k in know))
w2v = open('sgns.wiki.char', 'r')
w2v_filter = open('wiki_filtered', 'w')
V_filtered = open('V_filtered', 'w')
for line in w2v:
	if line.split()[0] in V:
		w2v_filter.write(line)
		V_filtered.write(line.split()[0]+'\n')
w2v.close()
w2v_filter.close()
V_filtered.close()

# 计算cosine相似度，找出与给定词语最相近的词
V_filter = [line.strip() for line in open('V_filtered', 'r')]
w2v = KeyedVectors.load_word2vec_format('wiki_filtered', binary=False)
for w in ['生物', '环境', '方法', '中国', '人类']:
	# w = random.sample(V_filter, 1)[0]
	result = w2v.similar_by_word(w, topn=10)
	print(result)

##################################################################################################
##                                           Task4                                              ##
##                 Computing semantic similarity using self-trained word embeddings             ##
##################################################################################################

# 设置参数
size = 100
min_count = 3
window = 1

# # 读取语料库，训练词向量
know_uncleaned = [str(line) for line in k_h['name'].tolist()]
know = [clean(k).split() for k in know_uncleaned]
model = Word2Vec(know, size=size, min_count=min_count, window=window)
model.save('w2v/{}_{}_{}'.format(size, min_count, window))

# 读取词向量模型
w2v = Word2Vec.load('w2v/{}_{}_{}'.format(size, min_count, window))

f = open('res_4', 'w')
# 计算cosine相似度，找出与给定词语最相近的词
V_filter = [line.strip() for line in open('V_filtered', 'r')]
# for i in range(5):
for w in ['生物', '环境', '方法', '中国', '人类']:
	# w = random.sample(V_filter, 1)[0]
	f.write(w)
	f.write('\n')
	result = w2v.similar_by_word(w, topn=10)
	for (word, sim) in result:
		f.write(word+', ')
	f.write('\n')

##################################################################################################
##                                           Task5                                              ##
##                           Sentence classification using Word2Vec                             ##
##################################################################################################

# 读取题目
tri_id = k_h[k_h['name'] == '三角函数与解三角形']['id'].values[0]
tri_family = np.append(k_h[k_h['root_id'] == tri_id]['id'].values, tri_id)
tri_qid = q_k_h_s[q_k_h_s['kh_id'].isin(tri_family)]['question_id'].values
tri_q_uncleaned = t_q_s[t_q_s['que_id'].isin(tri_qid)]['content'].tolist()
tri_q = [clean(q).split() for q in tri_q_uncleaned]

func_id = k_h[k_h['name'] == '函数与导数']['id'].values[0]
func_family = np.append(k_h[k_h['root_id'] == func_id]['id'].values, func_id)
func_qid = q_k_h_s[q_k_h_s['kh_id'].isin(func_family)]['question_id'].values
func_q_uncleaned = t_q_s[t_q_s['que_id'].isin(func_qid)]['content'].tolist()
func_q = [clean(q).split() for q in func_q_uncleaned]

questions = np.append(tri_q, func_q)

# 读取预训练词向量，生成句子向量
V_filter = [line.strip() for line in open('V_filtered_q', 'r')]
w2v = KeyedVectors.load_word2vec_format('wiki_filtered_q', binary=False)
X = np.zeros((39694, 300))
Y = []
i = 0
for q in questions:
	q_emb = []
	for word in q:
		try:
			q_emb.append(w2v[word])
		except KeyError:
			pass
	if len(q_emb) != 0:
		if i < len(tri_q):
			Y.append(0)
		else:
			Y.append(1)
		X[i, :]= np.mean(q_emb, axis=0)
		i += 1

# 划分训练集与测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)

# 模型训练及测试
clf = LinearSVC()
# clf = LogisticRegression()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

# 评估模型表现
print('Precision: ', precision_score(Y_test, Y_pred))
print('Recall: ', recall_score(Y_test, Y_pred))
print('F1 score: ', f1_score(Y_test, Y_pred))
print('Accuracy: ', accuracy_score(Y_test, Y_pred))
print('ROC_AUC score: ', roc_auc_score(Y_test, clf.decision_function(X_test)))
