# coding=utf-8
# ------本脚本从json或txt中获取文本内容，并合并行段落、分词、去除停用词------

import re
import os
import sys
import grpc
import json
#import jieba
import codecs
import datetime
import numpy as np
from tqdm import tqdm
from glob import glob
import hanlp_pb2
import hanlp_pb2_grpc
from pyhanlp import *

import csv
csv.field_size_limit(sys.maxsize)


def current_time():
    """
    获取当前时间：年月日时分秒
    :return:
    """
    ct = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    return ct


def hanlp_cut(texts):
    client = hanlp_pb2_grpc.GreeterStub(channel=grpc.insecure_channel("121.40.125.154:50051"))
    response = client.segment(hanlp_pb2.HanlpRequest(text=texts, indexMode=0, nameRecognize=1, translatedNameRecognize=1))
    return response.data


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    X, Y = [u'\u4e00', u'\u9fa5']  # unicode 前面加u
    if uchar >= X and uchar <= Y:
        return True
    else:
        return False


def is_number(uchar):
    """判断一个unicode是否是数字"""
    X, Y = [u'\u0030', u'\u0039']
    if uchar >= X and uchar <= Y:
        return True
    else:
        return False


def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    a, z, A, Z = [u'\u0041', u'\u005a', u'\u0061', u'\u007a', ]
    if (uchar >= a and uchar <= z) or (uchar >= A and uchar <= Z):
        return True
    else:
        return False


def is_other(uchar):
    """判断是否非汉字，数字和英文字符"""
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return True
    else:
        return False


def is_symbol(uchar):
    """判断是否中英文常用标点符号"""
    symbol_list = [u',', u'.', u'?', u'!', u'，', u'。', u'？', u'！', u'《', u'》',
                   u'（', u'）', u'：', u'、', u'“', u'”', u'；', '@', '/', '-',
                   '_', '(', ')']
    if uchar in symbol_list:
        return True
    else:
        return False


def separate_words(input_text, stopwords):
    out_text = []
    seg_list = jieba.cut(input_text)
    for word in seg_list:
        if is_other(word):
            continue
#        if is_number(word):
#            continue
        try:
            word = word.encode('UTF-8')
        except:
            continue
        if word not in stopwords:
            if word != ' ':
                out_text.append(word)
                out_text.append(' ')
    return out_text


def separate_words_by_hanlp(input_text, stopwords):
    out_text = []
    seg_list = hanlp_cut(input_text)  # 通过grpc调用hanlp
#    seg_list = HanLP.segment(input_text)  # 直接调用pyhanlp
    for term in seg_list:
        word = term.word
        if is_other(word):
            continue
        if is_number(word):
            continue
        if is_alphabet(word):
            continue
        try:
            word = word.encode('UTF-8')
        except:
            continue
        if word not in stopwords:
            if word != ' ':
                out_text.append(word)
                out_text.append(' ')
    return out_text


def get_font_size_mean(r):
    """获取json首页的字号平均值"""
    fonts_sum = 0
    cont_sum = 0
    for key, val in enumerate(r):
        if val.get('pageIndex') == 0:
            cont_texts = val.get('texts', [])
            if cont_texts is not []:
                cont_sum = cont_sum + len(cont_texts)
                for key2, val2 in enumerate(cont_texts):
                    fonts = val2.get('font_size')
                    fonts_sum = fonts_sum + fonts
    if cont_sum == 0:
        cont_sum = 1
    fonts_mean = float(fonts_sum) / float(cont_sum)
    return fonts_mean


if __name__ == '__main__':
#    base_dir = '/home/zhwpeng/abc/text_classify/data/0412/'
    # base_dir = '/home/abc/pzw/nlp/data/0412/'
    base_dir = '/home/abc/pzw/nlp/data/txt_json/'
    # 原始数据存放位置
    # base_data_dir = os.path.join(base_dir, 'raw/')
    base_data_dir = os.path.join(base_dir, 'train_data/')
#    base_data_dir = os.path.join(base_dir, 'test_data3/')
    # 分词后的数据存放位置
    separated_word_file_dir = os.path.join(base_dir, 'word_sep_ht03/')
    if not os.path.exists(separated_word_file_dir):
        os.makedirs(separated_word_file_dir)
    # 在txt中添加文本类型后csv存放位置
    csv_dataset_dir = os.path.join(base_dir, 'csv_dataset_ht03/')
    if not os.path.exists(csv_dataset_dir):
        os.makedirs(csv_dataset_dir)
    # 训练和测试的csv
#    train_test_csv_dir = os.path.join(base_dir, 'train_test_csv/')
    train_test_csv_dir = os.path.join(base_dir, 'train_csv_ht03/')
    if not os.path.exists(train_test_csv_dir):
        os.makedirs(train_test_csv_dir)

#    types = [
#             'NOTICE',
#             'T004001001', 'T004001002', 'D001002001', 'D001002002', 'D001002003', 'T004021008',
#             'D001003001', 'D001003002', 'D001003003', 'T004022018', 'D001004001', 'D001004002', 'D001004003',
#             'T004023007', 'T004004002', 'T004004005', 'T004004001', 'T004004004', 'T004004003', 'T004019001',
#             'T004019003',
#             'OTHER'
#            ]
    types = [
#             'NOTICE2',
             'NOTICE',
             'T004001001', 'T004001002', 'D001002001', 'D001002002', 'D001002003', 'T004021008',
             'D001003001', 'D001003002', 'D001003003', 'T004022018', 'D001004001', 'D001004002', 'D001004003',
             'T004023007', 'T004004002', 'T004004005', 'T004004001', 'T004004004', 'T004004003', 'T004019001',
             'T004019003',
             'OTHER',
             'T004009001', 'T004009002', 'T004005001', 'T004005002', 'T004006001',
             'T004006005'
            ]

    # types = ['110100', '110200', '110300', '110400', '110500', '110600', '110700', '110800', '210100', '210200',
    #          '210300', '210400', '220100', '220200', '220300', '220400', '220500', '220600', '230100', '240200',
    #          '240300', '240400', '240500', '270100', '270200', '270300', '270400', '270500', '280100', '280200',
    #          '280300', '280400', '330100', '330200', '340300', '340400', '350100', '350200', '360100', '360200',
    #          '360300', '360400', '370100', '370200', '370300', '370400', '370500', '370600', '410100', '410200',
    #          '410300', '410400', '420100', '420200', '420300', '420400', '420500', '420600', '420700', '420800',
    #          '430100', '430200', '450200', '450300', '450400', '450500', '460100', '460200', '460300', '460400',
    #          '460500', '480100', '490100', '490200', '490300', '510100', '610100', '610200', '610300', '620100',
    #          '620200', '620300', '620400', '620500', '630100', '630200', '630300', '630400', '640100', '640200',
    #          '640300', '640400', '640500', '650100', '650200', '650300', '650400', '710100', '710200', '720100',
    #          '720200', '720300', '730100', '730200']

    if 0:
        print ("第一步：文本去掉换行符、分词、去除停用词，保存文本内容至txt文件中...")
        print 'current time is', current_time()
        stop_words_file = "stopwords.txt"
        stopwords = []
        # 获取停用词
        with open(stop_words_file) as fi:
            for line in fi.readlines():
                stopwords.append(line.strip())
        print current_time()
        # jieba.load_userdict('abcChinese_jieba.txt')  # 引用自定义词库
#        jieba.load_userdict('abcChinese.txt')  # 引用自定义词库
#        user_words = ['行业研究', '存管制度', '实体经济']
#        for word in user_words:
#            jieba.add_word(word)

        do_txt, do_json = 1, 0
        # 处理txt
        if do_txt:
            for ty in tqdm(types):
                txt_dirs = glob(base_data_dir + ty + '/*.txt')
                output_dir = os.path.join(separated_word_file_dir, ty)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                for txt_dir in txt_dirs:
#                    print txt_dir.split('/')[-1]
                    all_words = []
                    with codecs.open(txt_dir, 'rb') as f:
                        r = f.read()
                    r = r.replace('\n', '').replace(' ', '')  # 段落合并、行合并
                    r = re.sub("[\s+：（）“”，■？、！…,/'《》<>!?_——=()-]+".decode("utf8", 'ignore'), "".decode("utf8", 'ignore'), r.decode("utf8", 'ignore'))
                    for li in r.split(u'。'):
                        lin = separate_words_by_hanlp(li, stopwords)
                        if len(lin):
                           for key, w in enumerate(lin):
                               if w != ' ':
                                   all_words.append(w)
                    # out_te = separate_words_by_hanlp(r, stopwords)
                    # print out_te
                    # if len(out_te):
                        # for key, w in enumerate(out_te):
                            # if w != ' ':
                                # if key < 800:
                                # all_words.append(w)
                    fulltext = ' '.join(all_words)
                    with codecs.open(output_dir + '/' + txt_dir.split('/')[-1], 'wb') as ft:
                        ft.write(fulltext)
        # 处理json
        if do_json:
            for folder in tqdm(types):  # 遍历所有类别文件夹，27个类别
                json_f = glob(base_data_dir + folder + '/*.json')
                dst_folder = os.path.join(separated_word_file_dir, folder)
                if not os.path.exists(dst_folder):
                    os.makedirs(dst_folder)
                for jf in json_f:  # 遍历所有文本，文本数不均衡，设置参数
                    all_words = []
                    with open(jf, 'rb') as f:
                        json_content = json.load(f)
                    if not len(json_content):
                        print 'json is None! skip!'
                        continue
                    font_size_mean = get_font_size_mean(json_content)  # 首页所有文字的平均字体大小
                    if font_size_mean == 0:
                        print 'no page 0 or no font size! skip!'
                        continue
                    all_text = ""
                    for val in json_content:
                        if val.get('pageIndex') == 0:
                            para_text = ""
                            cont_texts = val.get('texts', [])
                            if len(cont_texts):
                                for val2 in cont_texts:
                                    font_size = val2.get('font_size')  # 获取文字的字体大小
                                    text_ = val2.get('text')
                                    if font_size / font_size_mean <= 1.2:
                                        theta_fs = 1
                                    else:
                                        theta_fs = 4  # 文字权重（频率）系数（跟字体大小成正比）
                                    font_p = np.round(theta_fs * font_size / font_size_mean).astype('int')
                                    for fp in range(font_p):  # 根据文字权重值，重复该文字
                                        #text_ += text_
                                        para_text += text_
                                all_text += para_text
                    out_te = separate_words_by_hanlp(all_text, stopwords)
                    if len(out_te):
                        for w in out_te:
                            if w != ' ':
                                all_words.append(w)
                    fulltext = ' '.join(all_words)
                    # print fulltext
                    with open(dst_folder + '/' + jf.split('/')[-1].split('.')[0] + '.txt', 'w') as f:
                        f.write(fulltext)
        print 'current time is', current_time()


    if 1:
        print ("第二步：保存数据到csv文件中")
        print 'current time is', current_time()
        for ty in tqdm(types):
            path = os.path.join(csv_dataset_dir, ty + '.csv')
            csvfile = open(path, 'w')
            fieldnames = ['type', 'content']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            txt_dirs = glob(separated_word_file_dir + ty + '/*')
            for i in txt_dirs:
                label = os.path.dirname(i).split('/')[-1]
                content = open(i).read()
                writer.writerow({'type': label, 'content': content})
            csvfile.close()
        print 'current time is', current_time()

    if 1:
        print ("第三步：随机抽样，获取训练集和测试集")
        print 'current time is', current_time()
        csv_train = open(os.path.join(train_test_csv_dir, "train.csv"), 'a')
        csv_test = open(os.path.join(train_test_csv_dir, "test.csv"), 'a')
        fieldnames = ['type', 'content']
        writer_tr = csv.DictWriter(csv_train, fieldnames=fieldnames)
        writer_te = csv.DictWriter(csv_test, fieldnames=fieldnames)
        writer_tr.writeheader()
        writer_te.writeheader()

        cfs = glob(csv_dataset_dir + '/*.csv')
        for cf in cfs:
            print 'extracting file:', cf.split('/')[-1]
            try:
                cfi = open(cf, 'r')
                reader = csv.DictReader(cfi)
                for r in reader:
                    if np.random.random() <= 0.1:
                    # if 1:
                        writer_te.writerow({'type': r['type'], 'content': r['content']})
                    else:
                        writer_tr.writerow({'type': r['type'], 'content': r['content']})
            except:
                print "IO ERROR"
        print 'current time is', current_time()


