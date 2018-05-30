# coding=utf-8
# ------本脚本从txt中获取文本内容，并合并行段落、分词、去除停用词------

from __future__ import unicode_literals

import re
import os
import sys
import json
import jieba
import jieba.posseg as pseg
import codecs
import datetime
import numpy as np
from tqdm import tqdm
from glob import glob
from pyhanlp import *

nature = 'n nr ns nt nz nl ng t tg s f vi vg a ad an ag d i'
nature2 = 'eng m p pba pbei c cc u e y o w wkz wky wyz wyy wj ww wt wd wf wn wm ws wp wb wh'


def current_time():
    """
    获取当前时间：年月日时分秒
    :return:
    """
    ct = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    return ct


def separate_words_hanlp(input_text, stopwords=None):
    out_txt = list()
    # seg_list = jieba.cut(input_text)
    # seg_list = pseg.cut(input_text)
    # attachThreadToJVM()

    seg_list = HanLP.segment(input_text)
    for term in seg_list:
    # for word in seg_list:
    # for word, flag in seg_list:
        # print word, flag
        if term.word not in stopwords and term.word.strip() != '' and term.word.lower().isalpha():
        # if flag not in nature2.encode('utf-8').split(' ') and len(word.decode('utf-8')) > 1:
        # if len(word.decode('utf-8')) > 1:
        #     print word, flag
            out_txt.append(term.word)
    return ' '.join(out_txt[:400])


def separate_words_jieba(input_text, stopwords=None):
    out_txt = list()
    seg_list = jieba.cut(input_text)
    # seg_list = pseg.cut(input_text)
    # seg_list = HanLP.segment(input_text)
    for word in seg_list:
        if word not in stopwords and word.strip() != '' and word.lower().isalpha():
            out_txt.append(word)
    return ' '.join(out_txt[:600])


def separate_words_pseg(input_text, stopwords=None):
    out_txt = list()
    # seg_list = jieba.cut(input_text)
    seg_list = pseg.cut(input_text)
    # seg_list = HanLP.segment(input_text)
    # for term in seg_list:
    # for word in seg_list:
    for word, flag in seg_list:
        # print word, flag
        # if flag in nature.split(' ') and len(word) > 1 and word not in stopwords:
        # if flag not in nature2.encode('utf-8').split(' ') and len(word.decode('utf-8')) > 1:
        # if len(word.decode('utf-8')) > 1:
        #     print word, flag
        #if word not in stopwords and word.strip() != '' and ( word.lower().isalnum() or not word.lower().isdigit() or word.lower().isalpha()):
        if word not in stopwords and word.strip() != '' and word.lower().isalpha():
            out_txt.append(word.strip())
    tmp = ' '.join(out_txt[:400])
    return tmp


def pre_process(full_text, stopwords=None):
    full_text = full_text.replace(u'\n', '')
    full_text2 = separate_words_jieba(full_text, stopwords=stopwords)
    return full_text2


if __name__ == '__main__':
    jieba.load_userdict('abcChinese_j052301.txt')
    # base_dir = '/home/abc/ssd/pzw/nlp/data/'
    # base_dir = '/home/zhwpeng/abc/text_classify/data/0412/'
    base_dir = '/home/abc/ssd/pzw/nlp/data/data_industry/'
    # 原始数据存放位置
    # base_data_dir = os.path.join(base_dir, 'train_data/')
    # base_data_dir = os.path.join(base_dir, 'raw/train_data/')
    base_data_dir = os.path.join(base_dir, 'raw/')
    # 分词后的数据存放位置
    # separated_word_file_dir = os.path.join('/home/abc/ssd/pzw/nlp/data/0523/', 'word_sep/')
    # separated_word_file_dir = os.path.join('/home/zhwpeng/abc/text_classify/word2vec/', 'word_sep/')
    separated_word_file_dir = os.path.join(base_dir, 'word_sep_0528/')
    if not os.path.exists(separated_word_file_dir):
        os.makedirs(separated_word_file_dir)

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

    stop_words_file = "stop_words_v0520.txt"
    stop_words = set()
    # 获取停用词
    with codecs.open(stop_words_file, 'r', encoding='utf-8') as fi:
        for line in fi.readlines():
            stop_words.add(line.strip())
    # print stopwords

    if 1:
        for ty in tqdm(types):
            txt_dirs = glob(base_data_dir + ty + '/*.txt')
            output_dir = os.path.join(separated_word_file_dir, ty)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for tk, txt_dir in enumerate(txt_dirs):
                # print txt_dir
                try:
                    with codecs.open(txt_dir, 'r', encoding='utf-8') as f:
                        r = f.read()
                except:
                    print "cannot open {},maybe not utf-8 encoded".format(txt_dir)
                    continue
                # print r
                fulltext = pre_process(r, stopwords=stop_words)
                # print fulltext
                with codecs.open(output_dir + '/' + txt_dir.split('/')[-1], 'w',
                                 encoding='utf-8') as fw:
                    fw.write(fulltext)
