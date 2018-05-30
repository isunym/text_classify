# coding=utf-8
# ------本脚本用于按批获取文本的TF,IDF------

from __future__ import unicode_literals

import os
import re
import six
import codecs
import json
import datetime
import numpy as np
import pandas as pd
from math import log
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from pyhanlp import *
from glob import glob
from tqdm import tqdm

import csv
csv.field_size_limit(sys.maxsize)


def current_time():
    ct = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    return ct


def word_in_file_counts(word, batch_corpus):
    count = 0
    for cor in batch_corpus:
        for doc in cor:
            if doc.get(word):
                count += 1
            else:
                continue
    return count


def word_doc_counts(full_text):
    """统计词数，文本数"""
    word_tf = dict()
    doc_co = dict()
    for word in full_text:
        word_tf[word] = word_tf.get(word, 0.) + 1.0  # 统计词在文本中出现的频率（频数）
        doc_co[word] = 1.0  # 统计出现该词的文本数
    return word_tf, doc_co


def co_mul(a, b):
    for i, row in enumerate(a):
        a[i] = row * b[i]
    return a


def cal_idf(a, doc_num=272.):
    return [log((1. + doc_num) / (float(i) + 1.)) + 1 for i in a]


def ro_mul(a, b):
    """tfidf = TF * IDF"""
    for i, row in enumerate(a):
        a[i] = np.multiply(row, b)
    return a


def normalize(arr1):
    arr2 = np.multiply(arr1, arr1)
    norm = np.sqrt(np.sum(arr2, axis=1))
    arr_norm = np.array([x/norm[i] for i, x in enumerate(arr1)], float)
    return arr_norm


def get_y(fname):
    if fname in Noti:
        return 0
    elif fname in T411:
        return 1
    elif fname in T412:
        return 2
    elif fname in D121:
        return 3
    elif fname in D122:
        return 4
    elif fname in D123:
        return 5
    elif fname in T4218:
        return 6
    elif fname in D131:
        return 7
    elif fname in D132:
        return 8
    elif fname in D133:
        return 9
    elif fname in T42218:
        return 10
    elif fname in D141:
        return 11
    elif fname in D142:
        return 12
    elif fname in D143:
        return 13
    elif fname in T4237:
        return 14
    elif fname in T442:
        return 15
    elif fname in T445:
        return 16
    elif fname in T441:
        return 17
    elif fname in T444:
        return 18
    elif fname in T443:
        return 19
    elif fname in T4191:
        return 20
    elif fname in T4193:
        return 21
    elif fname in T491:
        return 22
    elif fname in T492:
        return 23
    elif fname in T451:
        return 24
    elif fname in T452:
        return 25
    elif fname in T461:
        return 26
    elif fname in T465:
        return 27
    elif fname in Othe:
        return 28


if __name__ == '__main__':
    # pd.set_option('max_colwidth', 200)
    # types = [
    #     'D001002001', 'D001002002', 'D001002003', 'D001003001', 'D001003002', 'D001003003', 'D001004001',
    #     'D001004002', 'D001004003',
    #     'NOTICE', 'OTHER',  # 去掉了NOTICE、OTHER
    #     'T004001001', 'T004001002', 'T004004001',
    #     'T004004002', 'T004004003', 'T004004004', 'T004004005', 'T004005001', 'T004005002', 'T004006001',
    #     'T004006005', 'T004009001', 'T004009002', 'T004019001', 'T004019003', 'T004021008', 'T004022018',
    #     'T004023007'
    # ]
    types = [
        '110100', '110200', '110300', '110400', '110500',
        '110600', '110700', '110800', '210100', '210200',
        '210300', '210400', '220100', '220200', '220300',
        '220400', '220500', '220600', '230100', '240200',
        '240300', '240400', '240500', '270100', '270200',
        '270300', '270400', '270500', '280100', '280200',
        '280300', '280400', '330100', '330200', '340300',
        '340400', '350100', '350200', '360100', '360200',
        '360300', '360400', '370100', '370200', '370300',
        '370400', '370500', '370600', '410100', '410200',
        '410300', '410400', '420100', '420200', '420300',
        '420400', '420500', '420600', '420700', '420800',
        '430100', '430200', '450200', '450300', '450400',
        '450500', '460100', '460200', '460300', '460400',
        '460500', '480100', '490100', '490200', '490300',
        '510100', '610100', '610200', '610300', '620100',
        '620200', '620300', '620400', '620500', '630100',
        '630200', '630300', '630400', '640100', '640200',
        '640300', '640400', '640500', '650100', '650200',
        '650300', '650400', '710100', '710200', '720100',
        '720200', '720300', '730100', '730200', '0'
    ]
    # base_data_dir = "/home/zhwpeng/abc/text_classify/data/0412/raw/train_data/"
    base_data_dir = "/home/zhwpeng/abc/nlp/data/report_industry/"
    # base_data_dir = "/home/abc/pzw/nlp/data/txt_json/train_data/"  # 10.15.10.10
    # base_data_dir = "/home/abc/ssd/pzw/nlp/data/train_data/"  # 10.15.8.8
    # dst_dir = "0520/051803/"
    dst_dir = "0530/"
    # dst_dir = "/home/abc/ssd/pzw/nlp/data/0518/"
    # if not os.path.exists(dst_dir+'full_contents/'):
    #     os.makedirs(dst_dir+'full_contents/')
    #     os.makedirs(dst_dir+'word_counts/')
    #     os.makedirs(dst_dir+'doc_counts/')

    if 1:
        print "按批获取文本，以及文本的统计数据(每批中关键词在每个文档中出现的次数，每批中关键词出现的文档数量)..."
        all_file_dirs = []
        for folder in types:
            for i, file_dir in enumerate(glob(base_data_dir+folder+'/*.txt')):
                if i > 30:  # 每个类别选20个样本
                    continue
                all_file_dirs.append(file_dir)
        print len(all_file_dirs)
        np.random.shuffle(all_file_dirs)  # 先打乱所有的训练集文件目录
        # print "打乱后所有的训练集文件目录是:", all_file_dirs

        print current_time()
        # 按批的方式取数据，并做处理
        nub = 0
        batch_size = 4
        for start in range(0, len(all_file_dirs), batch_size):
            print 'preprocessing texts batch {}, wait minutes...'.format(nub)
            word_count = dict()
            doc_count = dict()
            id_batch = list()
            full_txt = list()  # 每个id对应的全文，组成了一批的全文
            full_content = dict()
            end = min(start + batch_size, len(all_file_dirs))
            dirs_batch = all_file_dirs[start:end]
            for nums, txt_dir in enumerate(dirs_batch):
                with codecs.open(txt_dir, 'rb', encoding='UTF-8', errors='replace') as f:
                    txt_content = f.read()
                if not len(txt_content):
                    print 'txt contents is None! skip!'
                    continue
                # fu_txt = pre_process(txt_content)
                if not len(txt_content):
                    print 'maybe english file,skip!'
                    continue
                # print fu_txt
                txt_words = txt_content.split(' ')
                # txt_words = fu_txt.split(' ')[:100]  # 取100个词
                word_tf, word_co = word_doc_counts(txt_words)  # 统计文本中所有词的频数,包含该词的文本数
                # word_co = doc_counts(txt_words)  # 文本中出现的词设置为1
                # for w in word_c.keys():
                #     print w, word_c[w]
                word_count[txt_dir.split('/')[-1]] = word_tf
                doc_count[txt_dir.split('/')[-1]] = word_co
                # full_txt.append(' '.join(txt_words))
                # id_batch.append(txt_dir.split('/')[-1])
            # 为了和scikit-learn作对比，保存文档对应的内容
            # full_content['id'] = id_batch
            # full_content['content'] = full_txt
            # df = pd.DataFrame(full_content)
            # df.to_csv(dst_dir+'full_contents/'+'full_content_'+str(nub)+'.csv')
            df = pd.DataFrame(word_count)
            df = df.T
            df.to_csv(dst_dir+'word_counts/'+'word_count_'+str(nub)+'.csv')
            df = pd.DataFrame(doc_count)
            df = df.T
            # df.drop([df.columns[0]], axis=1, inplace=True)  # 去掉文件名
            df.loc['row_sum'] = df.apply(lambda x: x.sum())  # 统计词出现的文档数
            df.iloc[-1].to_csv(dst_dir+'doc_counts/'+'doc_count_'+str(nub)+'.csv', header=['word_counts'])
            print 'all files saved successfully!'
            nub += 1
            print current_time()

    if 0:
        print "按批计算文本的词频，选取每批按照词频排序的前10000个特征词，将每批的特征词保存为txt..."

        dst_dir1 = dst_dir + 'feature_words/'
        dst_dir2 = dst_dir + 'selected_tf/'
        if not os.path.exists(dst_dir1):
            os.makedirs(dst_dir1)
            os.makedirs(dst_dir2)
        # nub = 0
        # 计算tf
        for nub in range(14):
            df = pd.read_csv(dst_dir+'word_counts/'+'word_count_'+str(nub)+'.csv', low_memory=False)
            print df.columns[1:]
            # a = df.columns[1:].tolist()
            # print len(a), a[0]
            # # print df['Unnamed: 0']
            print df.drop(columns=['Unnamed: 0']).values.shape

            # 按频率筛选维度
            df01 = 1. / df.sum(axis=1)
            print df01.values.shape
            tf_df = co_mul(df.drop(columns=['Unnamed: 0']).values, df01.values)
            tf_df = pd.DataFrame(data=tf_df, index=df['Unnamed: 0'], columns=df.columns[1:])
            tf_df_sort = tf_df.sum(axis=0).sort_values(ascending=False)[:10000].index  # tfidf值降序取前10000
            with codecs.open(dst_dir1+'feature_words_'+str(nub)+'.txt', 'w') as fw:
                fw.write(' '.join(tf_df_sort.tolist()))
            print "feature_words saved successfully!"
            # print len(tf_df_sort.tolist())

            # tf_df = tf_df.ix[:, tf_df_sort]  # 筛选特征
            # # print ' '.join(tf_df_sort.tolist())
            # tf_df.to_csv(dst_dir+'tf_df_sorted.csv')
            # print "sorted tf csv files saved successfully!"

            # df = pd.DataFrame(data=df.drop(columns=['Unnamed: 0']).values,
            #                   index=df['Unnamed: 0'], columns=df.columns[1:])
            # df = df.ix[:, tf_df_sort]  # 筛选特征
            # # print ' '.join(tf_df_sort.tolist())
            # df.to_csv(dst_dir2 + 'tf_sorted_'+str(nub)+'.csv')
            # print "sorted tf csv files saved successfully!"

    if 0:
        print '每批筛选出来的关键词做并集,并保存为txt(tf和idf统一要用的，并且测试集也要以这份特征词为准)...'
        dst_dir1 = dst_dir + 'feature_words/'
        words_all = set()
        for i in range(14):
            with codecs.open(dst_dir1+'feature_words_'+str(i)+'.txt', 'r',
                             encoding='utf-8', errors='replace') as fr:
                tf_sort = fr.read()
            words_batch = tf_sort.split(' ')
            print '第{}批的关键词数量是{}'.format(i, len(words_batch))
            for w in words_batch:
                words_all.add(w)
        print '所有批关键词合并后的数量是{}'.format(len(words_all))

        # ******与词库查看重合度******
        with codecs.open(dst_dir + 'feature_words_v0520.txt', 'r',
                         encoding='utf-8', errors='replace') as fr:
            feature_words = fr.read()
        feature_vocab = set()
        feature_words = feature_words.splitlines()
        print '原词库中的词总数是', len(feature_words)
        for lin in feature_words:
            words = lin.split(' ')
            feature_vocab.add(words[0].decode('utf-8'))
        collect = feature_vocab & words_all  # 前面两个集合里词编码要一致，否则交集为0
        print '重合的特征词总数是', len(collect)

        words_all = list(words_all)
        with codecs.open(dst_dir+'features_merged.txt', 'w') as fw:
            fw.write(' '.join(words_all))
        print "features_merged saved successfully!"

    if 1:
        print '查看分词后的文本特征词与词库(基于规则)的重合度...'

        feature_words_from_df = set()
        # for i in range(14):
        #     feature_words_df = pd.read_csv(dst_dir + 'doc_counts/doc_count_'+str(i)+'.csv', low_memory=False)
        #     # print feature_words_df.values[:, 0].tolist()
        #     # print len(feature_words_df.values[:, 0].tolist())
        #     for word in feature_words_df.values[:, 0].tolist():
        #         feature_words_from_df.add(word.decode('utf-8'))
        # # print feature_words_from_df
        # print '文本分词后的特征词数量是', len(feature_words_from_df)

        feature_words_df = pd.read_csv(dst_dir + 'doc_counts/doc_count_' + str(00) + '.csv', low_memory=False)
        for word in feature_words_df.values[:, 0].tolist():
            feature_words_from_df.add(word.decode('utf-8'))

        print '文本分词后的特征词数量是', len(feature_words_from_df)
        with codecs.open(dst_dir + 'feature_words_v0522.txt', 'r',
                         encoding='utf-8', errors='replace') as fr:
            feature_words = fr.read()
        feature_vocab = set()
        feature_words = feature_words.splitlines()
        print '原词库中的词总数是', len(feature_words)
        for lin in feature_words:
            words = lin.split(' ')
            feature_vocab.add(words[0].decode('utf-8'))
        print len(feature_vocab)
        collect = feature_vocab & feature_words_from_df  # 前面两个集合里词编码要一致，否则交集为0
        print '重合的特征词总数是', len(collect)

        # with codecs.open(dst_dir+'collection_words.txt', 'w') as fw:
        #     fw.write(' '.join(list(collect)))
        # with codecs.open(dst_dir+'unc.txt', 'w') as fw:
        #     fw.write(' '.join(list(feature_vocab-collect)))
        # print 'feature words saved successfully!'

    if 0:
        print '利用合并后的关键词筛选doc counts...'
        with codecs.open(dst_dir+'features_merged.txt', 'r') as fr:
            words_all = fr.read()
        words_all = words_all.split(' ')  # 注意此处的编码要跟后面的一致，与csv文档要一致
        # print words_all
        for i in range(14):
            df = pd.read_csv(dst_dir+'doc_counts/'+'doc_count_'+str(i)+'.csv', low_memory=False)
            # print df.values[:, 0].tolist()
            doc_df = pd.DataFrame(data=df.values[:, -1], index=df.values[:, 0], columns=df.columns[1:])
            # words_set = set()
            # for word in df.values[:, 0].tolist():
            #     words_set.add(word)
            # doc_df = doc_df.ix[list(set(df.values[:, 0].tolist()) & set(words_all))]  # 两个集合的元素编码要保持一致，不能用utf-8
            # doc_df = doc_df.ix[words_set]
            doc_df = doc_df.ix[words_all]
            doc_df.to_csv(dst_dir+'doc_counts_fit/doc_counts_s_'+str(i)+'.csv')
            print 'doc counts fit selected successfully!'
    if 0:
        print '将每批的doc counts相加，得到所有文档的关键词'
        with codecs.open(dst_dir+'features_merged.txt', 'r') as fr:
            words_all = fr.read()
        words_all = words_all.split(' ')  # 注意此处的编码要跟后面的一致，与csv文档要一致

        doc_c = np.zeros(18910, 'float')
        # print doc_c.shape
        for i in range(14):
            df = pd.read_csv(dst_dir+'doc_counts_fit/doc_counts_s_'+str(i)+'.csv', low_memory=False)
            df.fillna(value=0, inplace=True)  # 有些批里不含有一些关键词，其统计值为空，需要先填充0
            print df.values[:, 1]
            doc_c = np.add(doc_c, df.values[:, 1])
            index = df.values[:, 0]
            columns = df.columns[1:]
        print doc_c.shape
        doc_c_all = pd.DataFrame(data=doc_c, index=words_all, columns=['word_counts'])
        doc_c_all.to_csv(dst_dir+'doc_c_all.csv')
        print 'doc_c_all saved successfully!'

    if 0:
        print "将特征词统一到每批的词频统计里"

        with codecs.open(dst_dir + 'features_merged.txt', 'r') as fr:
            words_all = fr.read()
        words_all = words_all.split(' ')  # 注意此处的编码要跟后面的一致，与csv文档要一致
        # print words_all
        for i in range(14):
            df = pd.read_csv(dst_dir + 'word_counts/' + 'word_count_' + str(i) + '.csv', low_memory=False)
            doc_df = pd.DataFrame(data=df.drop(columns=['Unnamed: 0']).values, index=df['Unnamed: 0'],
                                  columns=df.columns[1:])
            doc_df = doc_df.ix[:, words_all]
            doc_df.to_csv(dst_dir + 'word_counts_fit/word_counts_s_' + str(i) + '.csv')
            print 'word counts fit selected saved successfully!'

    if 0:
        print "计算特征词的IDF..."

        doc_num = 54543
        select = 0
        # 计算idf,所有批次共用一套IDF
        df = pd.read_csv(dst_dir+'doc_c_all.csv', low_memory=False)
        # print df.columns
        # print df.values
        # a = df.values[:, -1]
        # print a, a.shape
        idf_df = cal_idf(df.values[:, -1], doc_num=doc_num)
        idf_df = pd.DataFrame(data=idf_df, index=df.values[:, 0], columns=df.columns[1:])
        idf_df.to_csv(dst_dir+'idf_df.csv')
        print '特征词的IDF保存完成'

    if 0:
        print "按批计算TF-IDF..."

        # 计算tf-idf
        df = pd.read_csv(dst_dir+'idf_df.csv', low_memory=False)
        # print df.values[:, -1]
        print "特征词的idf维度:", df.values[:, -1].shape  # 特征词的idf维度

        for i in range(14):
            df02 = pd.read_csv(dst_dir+'word_counts_fit/word_counts_s_'+str(i)+'.csv', low_memory=False)
            index = df02['Unnamed: 0']
            columns = df02.columns[1:]
            # print df02['Unnamed: 0']
            print "文档的TF维度:", df02.drop(columns=['Unnamed: 0']).values.shape
            tf_idf = ro_mul(df02.drop(columns=['Unnamed: 0']).values, df.values[:, -1])  # 开始计算每批的tf-idf
            tfidf_df = pd.DataFrame(data=tf_idf, index=index, columns=columns)
            # tfidf_df.to_csv(dst_dir+'tfidf_df_'+str(i)+'.csv')  # 保存每批18910维tfidf

            # 特征值规范化
            tfidf_df.fillna(value=0, inplace=True)  # 缺失值填充
            # print tfidf_df.values
            tfidf_df = pd.DataFrame(data=normalize(tfidf_df.values), index=index, columns=columns)
            tfidf_df.to_csv(dst_dir + 'tfidf_df_normed_1/tfidf_df_norm_'+str(i)+'.csv')
            print 'TF-IDF 保存成功!'

            # 对TF-IDF排序
            tfidf_sorted = tfidf_df.sum(axis=0).sort_values(ascending=False)[1000:2000].index  # TF-IDF前1000维特征词***
            with codecs.open(dst_dir+'feature_words_iii/feature_words_'+str(i)+'.txt', 'w') as fw:
                fw.write(' '.join(tfidf_sorted.tolist()))
            print "feature_words saved successfully!"

    if 0:
        dst_dir3 = dst_dir + 'feature_words_ii/'
        words_all = set()
        for i in range(14):
            with codecs.open(dst_dir3 + 'feature_words_' + str(i) + '.txt', 'r') as fr:
                tf_sort = fr.read()
            words_batch = tf_sort.split(' ')
            print '第{}批的关键词数量是{}'.format(i, len(words_batch))
            for w in words_batch:
                words_all.add(w)
        print '所有批关键词合并后的数量是{}'.format(len(words_all))
        words_all = list(words_all)
        with codecs.open(dst_dir + 'features_merged_ii.txt', 'w') as fw:
            fw.write(' '.join(words_all))
        print "features_merged ii saved successfully!"

    if 0:
        print "将特征词统一到每批的TF-IDF里"

        with codecs.open(dst_dir + 'features_merged_ii.txt', 'r') as fr:
            words_all = fr.read()
        words_all = words_all.split(' ')  # 注意此处的编码要跟后面的一致，与csv文档要一致
        # print words_all
        for i in range(14):
            df = pd.read_csv(dst_dir + 'tfidf_df_normed/tfidf_df_norm_' + str(i) + '.csv', low_memory=False)
            doc_df = pd.DataFrame(data=df.drop(columns=['Unnamed: 0']).values, index=df['Unnamed: 0'],
                                  columns=df.columns[1:])
            doc_df = doc_df.ix[:, words_all]
            doc_df.to_csv(dst_dir + 'tfidf_fit/tfidf_s_' + str(i) + '.csv')
            print 'TF-IDF fit saved successfully!'

    if 0:
        print '将每批TF-IDF合并成一个文件'
        df00 = pd.read_csv(dst_dir + 'tfidf_fit/tfidf_s_' + str(0) + '.csv')
        df01 = pd.read_csv(dst_dir + 'tfidf_fit/tfidf_s_' + str(1) + '.csv')
        df02 = pd.read_csv(dst_dir + 'tfidf_fit/tfidf_s_' + str(2) + '.csv')
        df03 = pd.read_csv(dst_dir + 'tfidf_fit/tfidf_s_' + str(3) + '.csv')
        df04 = pd.read_csv(dst_dir + 'tfidf_fit/tfidf_s_' + str(4) + '.csv')
        df05 = pd.read_csv(dst_dir + 'tfidf_fit/tfidf_s_' + str(5) + '.csv')
        df06 = pd.read_csv(dst_dir + 'tfidf_fit/tfidf_s_' + str(6) + '.csv')
        df07 = pd.read_csv(dst_dir + 'tfidf_fit/tfidf_s_' + str(7) + '.csv')
        df08 = pd.read_csv(dst_dir + 'tfidf_fit/tfidf_s_' + str(8) + '.csv')
        df09 = pd.read_csv(dst_dir + 'tfidf_fit/tfidf_s_' + str(9) + '.csv')
        df10 = pd.read_csv(dst_dir + 'tfidf_fit/tfidf_s_' + str(10) + '.csv')
        df11 = pd.read_csv(dst_dir + 'tfidf_fit/tfidf_s_' + str(11) + '.csv')
        df12 = pd.read_csv(dst_dir + 'tfidf_fit/tfidf_s_' + str(12) + '.csv')
        df13 = pd.read_csv(dst_dir + 'tfidf_fit/tfidf_s_' + str(13) + '.csv')

        # add_df = pd.DataFrame(data=df00.append(df01.append(df02)), index=df00['Unnamed: 0'],
        #                       columns=df00.columns[1:])

        (
            df00.append(
                df01.append(
                    df02.append(
                        df03.append(
                            df04.append(
                                df05.append(
                                    df06.append(
                                        df07.append(
                                            df08.append(
                                                df09.append(
                                                    df10.append(
                                                        df11.append(
                                                            df12.append(
                                                                df13, ignore_index=True), ignore_index=True),
                                                        ignore_index=True), ignore_index=True), ignore_index=True),
                                            ignore_index=True),
                                        ignore_index=True),
                                    ignore_index=True),
                                ignore_index=True),
                            ignore_index=True),
                        ignore_index=True),
                    ignore_index=True),
                ignore_index=True)
        ).to_csv(dst_dir + 'tf_idf_all.csv', index=False)
        print 'TF-IDF added saved successfully!'

    if 0:
        print '获取标签'
        types = ['NOTICE', 'T004001001', 'T004001002', 'D001002001', 'D001002002', 'D001002003', 'T004021008',
                 'D001003001', 'D001003002', 'D001003003', 'T004022018', 'D001004001', 'D001004002', 'D001004003',
                 'T004023007', 'T004004002', 'T004004005', 'T004004001', 'T004004004', 'T004004003', 'T004019001',
                 'T004019003', 'T004009001', 'T004009002', 'T004005001', 'T004005002', 'T004006001', 'T004006005',
                 'OTHER']
        Noti = os.listdir(os.path.join(base_data_dir, types[0]))
        T411 = os.listdir(os.path.join(base_data_dir, types[1]))
        T412 = os.listdir(os.path.join(base_data_dir, types[2]))
        D121 = os.listdir(os.path.join(base_data_dir, types[3]))
        D122 = os.listdir(os.path.join(base_data_dir, types[4]))
        D123 = os.listdir(os.path.join(base_data_dir, types[5]))
        T4218 = os.listdir(os.path.join(base_data_dir, types[6]))
        D131 = os.listdir(os.path.join(base_data_dir, types[7]))
        D132 = os.listdir(os.path.join(base_data_dir, types[8]))
        D133 = os.listdir(os.path.join(base_data_dir, types[9]))
        T42218 = os.listdir(os.path.join(base_data_dir, types[10]))
        D141 = os.listdir(os.path.join(base_data_dir, types[11]))
        D142 = os.listdir(os.path.join(base_data_dir, types[12]))
        D143 = os.listdir(os.path.join(base_data_dir, types[13]))
        T4237 = os.listdir(os.path.join(base_data_dir, types[14]))
        T442 = os.listdir(os.path.join(base_data_dir, types[15]))
        T445 = os.listdir(os.path.join(base_data_dir, types[16]))
        T441 = os.listdir(os.path.join(base_data_dir, types[17]))
        T444 = os.listdir(os.path.join(base_data_dir, types[18]))
        T443 = os.listdir(os.path.join(base_data_dir, types[19]))
        T4191 = os.listdir(os.path.join(base_data_dir, types[20]))
        T4193 = os.listdir(os.path.join(base_data_dir, types[21]))
        T491 = os.listdir(os.path.join(base_data_dir, types[22]))
        T492 = os.listdir(os.path.join(base_data_dir, types[23]))
        T451 = os.listdir(os.path.join(base_data_dir, types[24]))
        T452 = os.listdir(os.path.join(base_data_dir, types[25]))
        T461 = os.listdir(os.path.join(base_data_dir, types[26]))
        T465 = os.listdir(os.path.join(base_data_dir, types[27]))
        Othe = os.listdir(os.path.join(base_data_dir, types[28]))
        df = pd.read_csv(dst_dir+'tf_idf_all.csv', low_memory=False)
        print df['Unnamed: 0']
        label = [get_y(txt_name) for txt_name in df['Unnamed: 0'].tolist()]
        print len(label)
        labeled_df = pd.DataFrame(data=df.drop(columns=['Unnamed: 0']).values, index=label,
                              columns=df.columns[1:])
        labeled_df.to_csv(dst_dir + 'tf_idf_all_labeled.csv')
        print 'TF-IDF fit saved successfully!'

    if 0:
        print '每批筛选出来的关键词做合并'
        with codecs.open(dst_dir+'features_by_tf_sorted_'+str(0)+'.txt', 'r') as fr:
            tf_sort_0 = fr.read()
        with codecs.open(dst_dir+'features_by_tf_sorted_'+str(1)+'.txt', 'r') as fr:
            tf_sort_1 = fr.read()
        feature0 = tf_sort_0.split(' ')
        feature1 = tf_sort_1.split(' ')
        print len(feature0), len(feature1)
        feature_all = list(set(feature0+feature1))
        print len(feature_all)
        with codecs.open(dst_dir+'features_merge_'+str(01)+'.txt', 'w') as fw:
            fw.write(' '.join(feature_all))
        print "features_mrege saved successfully!"

        df = pd.read_csv(dst_dir+'doc_counts/'+'doc_count_0.csv', low_memory=False)
        # print df.values[:, 0].tolist()
        doc_df = pd.DataFrame(data=df.values[:, -1], index=df.values[:, 0], columns=df.columns[1:])
        doc_df = doc_df.ix[list(set(df.values[:, 0].tolist()) & set(feature_all))]
        # doc_df = doc_df.ix[feature_all]
        doc_df.to_csv(dst_dir+'doc_counts_selected_0_.csv')
        print 'doc counts selected successfully!'

    if 0:
        print "计算特征词的IDF..."

        batch_size = 256
        select = 0
        # 计算idf
        df = pd.read_csv(dst_dir+'doc_counts/'+'doc_count_0.csv', low_memory=False)
        print df.columns
        # print df.values
        a = df.values[:, -1]
        print a, a.shape
        idf_df = cal_idf(df.values[:, -1], batch_size=batch_size)
        idf_df = pd.DataFrame(data=idf_df, index=df.values[:, 0], columns=df.columns[1:])
        # idf_df.to_csv(dst_dir+'idf_df.csv')
        if select:
            print '按照词频筛选'
            with codecs.open(dst_dir+'features_by_tf_sorted.txt', 'r') as fr:
                features_by_tf_sorted = fr.read()
            features_by_tf_sorted = features_by_tf_sorted.split(' ')
            print 'features_by_tf_sorted numbers', len(features_by_tf_sorted)
            idf_df_sort = idf_df.ix[features_by_tf_sorted]
            idf_df_sort.to_csv(dst_dir+'idf_df_sorted.csv')

    if 0:
        print "按批计算TF-IDF..."

        # 计算tf-idf
        df = pd.read_csv(dst_dir+'word_counts/word_count_0.csv', low_memory=False)
        # print df.columns
        print df['Unnamed: 0']
        print df.drop(columns=['Unnamed: 0']).values.shape

        df02 = pd.read_csv(dst_dir+'idf_df.csv', low_memory=False)
        print df02.values[:, -1].shape
        print df02.values[:, -1]
        tfidf_df = ro_mul(df.drop(columns=['Unnamed: 0']).values, df02.values[:, -1])
        tfidf_df = pd.DataFrame(data=tfidf_df, index=df['Unnamed: 0'], columns=df.columns[1:])
        # tfidf_df.to_csv(dst_dir+'tfidf_df.csv')

        tfidf_df.fillna(value=0, inplace=True)  # 缺失值填充
        print tfidf_df.values
        tfidf_df = pd.DataFrame(data=normalize(tfidf_df.values), index=df['Unnamed: 0'], columns=df.columns[1:])
        tfidf_df.to_csv(dst_dir+'tfidf_df_norm.csv')

    if 0:
        print '按照tfidf筛选'
        tfidf_df = pd.read_csv(dst_dir+'tfidf_df_.csv', low_memory=False)
        features_index_by_tfidf = tfidf_df.sum(axis=0).sort_values(ascending=False)[:100].index
        print features_index_by_tfidf
        count_v0_tfsx_df = tfidf_df.ix[:, features_index_by_tfidf]
        count_v0_tfsx_df.to_csv(dst_dir+'0516/'+'count_v0_tfsx_df02.csv')

    if 0:
        df = pd.read_csv(dst_dir+'full_contents/'+'full_content_0.csv')
        all_texts = df['content']
        count = CountVectorizer(decode_error='replace')
        counts_all = count.fit_transform(all_texts)
        tf_df = pd.DataFrame(counts_all.toarray(), columns=count.get_feature_names())
        tf_df.to_csv(dst_dir+'tf_df.csv')

        transformer = TfidfTransformer(norm='l2',)
        tfidf = transformer.fit_transform(counts_all)
        tfidf_df = pd.DataFrame(tfidf.toarray(), columns=count.get_feature_names())
        tfidf_df.to_csv(dst_dir+'tfidf_df.csv')
        # df = pd.read_csv(dst_dir+'tf_all.csv', low_memory=False)
        # print df.sum(axis=1)
        # df = pd.read_csv(dst_dir + 'word_counts/word_count_0.csv', low_memory=False)
        # print df.sum(axis=1)

