# coding=utf-8
# ------本脚本使用SVM训练文本分类器------

import os
import pickle
import datetime
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def current_time():
    ct = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    return ct


def get_non(df_file):
    """
    # 找到空行
    :param df_file:
    :return: vacant_row = [93, 1487, 1595]
    """
    vacant_row = []
    for ind, row in df_file.iterrows():
        if not isinstance(row['content'], str) or row['content'] == '\n':
            # print row
            vacant_row.append(ind)
    return vacant_row


def get_data(train_filename="/home/zhwpeng/abc/nlp/data/0321/csvfiles/cleaned_train2.csv",
             test_filename="/home/zhwpeng/abc/nlp/data/0321/csvfiles/cleaned_test.csv"):
    train_df = pd.read_csv(train_filename, compression=None, error_bad_lines=False)
    vr = get_non(train_df)
    if vr is not None:
        train_df.drop(vr, inplace=True)

    if test_filename is not "":
        test_df = pd.read_csv(test_filename, compression=None, error_bad_lines=False)
        vr_test = get_non(test_df)
        if vr_test is not None:
            test_df.drop(vr_test, inplace=True)
        content_df = train_df.append(test_df, ignore_index=True)
    else:
        content_df = train_df

    # shuffle data
    content_df = shuffle(content_df)

    all_texts = content_df['content']
    all_labels = content_df['type']

    print "沪深股研报文本数量：", len(all_texts), len(all_labels)
    print "每类研报的数量：\n", all_labels.value_counts()
    return all_texts, all_labels


def get_count(all_texts, max_features=10000, save=True, vocab_dir='train_tmp/all_features_dir/'):
    """训练集的关键词 vocabulary 需要保存或被引用"""
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)
    if not os.path.exists(vocab_dir + 'all_features.pkl'):
        count = CountVectorizer(decode_error='replace', max_features=max_features)
        counts_all = count.fit_transform(all_texts)
        print '所有的特征词维度是: ' + repr(counts_all.shape)
        if save:
            with open(vocab_dir + 'all_features.pkl', 'wb') as f:  # save vocabulary
                pickle.dump(count.vocabulary_, f)
            print 'all features saved successfully!'
    else:
        with open(vocab_dir + 'all_features.pkl', 'rb') as f:  # load vocabulary
            vocab = pickle.load(f)
        count = CountVectorizer(decode_error='replace', vocabulary=vocab)
        print 'all features loaded successfully!'
    return count


def get_tfidf(count_v0, train_texts):
    counts_train = count_v0.fit_transform(train_texts)

    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(counts_train)
    print 'The shape of TF-IDF is:', tfidf.shape

    feature_names = count_v0.get_feature_names()  # 关键字
    count_v0_df = pd.DataFrame(counts_train.toarray())
    tfidf_df = pd.DataFrame(tfidf.toarray())
    return count_v0_df, tfidf_df, feature_names


def guiyi(x):
    x[x > 1] = 1
    return x


def select_index_and_get_x(count_v0_df, tfidf_df, feature_names,
                           all_labels, tfidf_features=5000, chi2_features=500,
                           features_index_by_ch2_dir='train_tmp/features_index_by_ch2/'):
    """通过tfidf第一次降维，通过卡方检测第二次降维
    :param count_v0_df:
    :param tfidf_df:
    :param feature_names: 所有的特征值（关键词）
    :param all_labels:
    :param tfidf_features: tfidf之后的维度
    :param chi2_features: 卡方之后的维度
    :return:
    """
#    features_index_by_tfidf = tfidf_df.sum(axis=0).sort_values(ascending=False)[:tfidf_features].index  # tfidf值降序取前5000
    features_index_by_tfidf = tfidf_df.sum(axis=0).sort_values(ascending=False)[4000:(4000+tfidf_features)].index  # tfidf值降序取前5000
    print 'features_index_by_tfidf:', len(features_index_by_tfidf)

    count_v0_tfsx_df = count_v0_df.ix[:, features_index_by_tfidf]   # tfidf筛选后的词向量矩阵
    df_columns = pd.Series(feature_names)[features_index_by_tfidf]

    tfidf_df_1 = count_v0_tfsx_df.apply(guiyi)
    tfidf_df_1.columns = df_columns
    le = preprocessing.LabelEncoder()
    tfidf_df_1['label'] = le.fit_transform(all_labels)
    # le.fit(classes)
    # tfidf_df_1['label'] = le.transform(all_labels)
    ch2 = SelectKBest(chi2, k=chi2_features)
    nolabel_feature = [x for x in tfidf_df_1.columns if x not in ['label']]
    ch2_sx_np = ch2.fit_transform(tfidf_df_1[nolabel_feature], tfidf_df_1['label'])
    label_np = np.array(tfidf_df_1['label'])

    # 训练集的卡方筛选后的特征词的index保存，后面测试使用
    if not os.path.exists(features_index_by_ch2_dir):
        os.makedirs(features_index_by_ch2_dir)
    index_tf = np.array(features_index_by_tfidf)
    a = ch2.get_support()  # 卡方检测之后的index真假矩阵
    features_index_by_ch2 = []
    for ke, v in enumerate(a):
        if v:
            features_index_by_ch2.append(index_tf[ke])
    print '卡方检验选出的特征词', features_index_by_ch2
    ch_columns = pd.Series(feature_names)[features_index_by_ch2]
    key_words = ' '.join(ch_columns.tolist())
    print key_words
    np.save(features_index_by_ch2_dir + 'features_index_by_ch2.npy', np.array(features_index_by_ch2))
    print '卡方检验选出的特征词保存成功!'

    x, y = ch2_sx_np, label_np
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    return x, y


def get_x_test(count_v0_df, features_index_by_ch2_dir='train_tmp/features_index_by_ch2/'):
    """
    :param count_v0_df: 测试集的tfidf矩阵
    :param feature_names: 所有的特征词
    :param features_index_by_ch2_dir: 训练过程中保存的卡方检测筛选后的特征维度路径
    :return:
    """
    selected_features_index = np.load(features_index_by_ch2_dir + 'features_index_by_ch2.npy')  # 导入训练集tfidf筛选后的维度索引
    count_v0_tfsx_df = count_v0_df.ix[:, selected_features_index]  # tfidf法筛选后的词向量矩阵

    tfidf_df_1 = count_v0_tfsx_df.apply(guiyi)

    return tfidf_df_1


if __name__ == '__main__':
#    classes = ['NOTICE', 'T004001001', 'T004001002', 'S004001', 'D001002001', 'D001002002', 'D001002003',
#               'T004021008', 'S004021', 'D001003001', 'D001003002', 'D001003003', 'T004022018', 'S004022',
#               'D001004001', 'D001004002', 'D001004003', 'T004023007', 'S004023', 'T004004002', 'T004004005',
#               'T004004001', 'T004004004', 'T004004003', 'S004004', 'T004019001', 'T004019003', 'S004019',
#               'OTHER']
    train_flag = 1
    if train_flag:
        base_dir = '/home/abc/pzw/nlp/data/txt_json/'
#        base_dir = '/home/zhwpeng/abc/text_classify/data/0404/'
        print '(1) 数据准备...'
        print 'current time is', current_time()
        all_texts, all_labels = get_data(train_filename=base_dir+"train_csv_ht03/train.csv", test_filename="")
        print '(2) 计算词频、tfidf矩阵、卡方检验筛选特征维度...'
        train_tmp_dir = 'train_tmp_ht03/'
        print 'current time is', current_time()
        count_v0 = get_count(all_texts, max_features=20000, save=True,
                             vocab_dir=base_dir+train_tmp_dir+'all_features_dir/')
        count_v0_df, tfidf_df, feature_names = get_tfidf(count_v0, all_texts)
        ch = 1000
        X, y = select_index_and_get_x(count_v0_df, tfidf_df, feature_names,
                                      all_labels, tfidf_features=12000, chi2_features=ch,
                                      features_index_by_ch2_dir=base_dir+train_tmp_dir+'features_index_by_tf12000ch2_'+str(ch)+'/')
        print 'X shape:', X.shape
        print 'y shape:', y.shape

        print '(3) 训练和评估SVM分类器...'
        print 'current time is', current_time()
        kfold = 0
        if kfold:
            # 进行k折交叉验证
            print '进行k折交叉验证...'
            skf = StratifiedKFold(y, n_folds=5)
            y_pre = y.copy()
            for train_index, test_index in skf:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                svclf = SVC(kernel='linear')
                svclf.fit(X_train, y_train)
                y_pre[test_index] = svclf.predict(X_test)
            print '准确率为 %.6f' % (np.mean(y_pre == y))
            print y_pre.shape
            print 'current time is', current_time()

        if not kfold:
            # 不做k折交叉验证
            print '不做k折交叉验证...'
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
            tune = 0
            if not tune:
                # 不做超参数的调整，直接训练
                print '不做k折交叉验证，不做超参数的调整，直接按默认参数训练...'
                svclf = SVC(kernel='rbf', C=10, gamma=0.001, verbose=0)
                svclf2 = SVC(kernel='rbf', C=100, gamma=0.001, verbose=0)
                svclf.fit(x_train, y_train)
                svclf2.fit(x_train, y_train)
                print '分类器训练完毕。'
                print '(4) 在验证集上评价分类器...'
                print 'current time is', current_time()
                preds = svclf.predict(x_test)
                preds2 = svclf2.predict(x_test)
                num = 0
                num2 = 0
                preds = preds.tolist()
                preds2 = preds2.tolist()

                for i, pred in enumerate(preds):
                    if int(pred) == int(y_test[i]):
                        num += 1
                score = "%.4f"%(float(num) / len(preds))
                print 'SVM precision_score:' + str(score)
                print 'current time is', current_time()

                for i, pred2 in enumerate(preds2):
                    if int(pred2) == int(y_test[i]):
                        num2 += 1
                score2 = "%.4f"%(float(num2) / len(preds2))
                print 'SVM2 precision_score:' + str(score2)
                print 'current time is', current_time()

                if not os.path.exists(base_dir+train_tmp_dir+'new_model/'):
                    os.makedirs(base_dir+train_tmp_dir+'new_model/')
                with open(base_dir+train_tmp_dir+'new_model/svm_c29'+'ch'+str(ch)+'s'+str(score)+'.pkl', 'wb') as fw:
                    pickle.dump(svclf, fw)
                print '分类器模型文件保存成功!'

                with open(base_dir+train_tmp_dir+'new_model/svm_c29'+'ch'+str(ch)+'s'+str(score2)+'.pkl', 'wb') as fw2:
                    pickle.dump(svclf2, fw2)
                print '分类器模型文件保存成功!'
            else:
                # 调整超参数
                print '不做k折交叉验证，采用GridSearchCV调整超参数...'
                tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
                scores = ['precision', 'recall']
                for score in scores:
                    print("依据 %s 调整超参数" % score)
                    # 构造这个GridSearch的分类器,5-fold
                    clf = GridSearchCV(SVC(), tuned_parameters, cv=3, scoring='%s_weighted' % score, n_jobs=-1, verbose=10)
                    # 只在训练集上面做k-fold,然后返回最优的模型参数
                    clf.fit(x_train, y_train)
                    print("已找到最优超参数，打印参数如下:")
                    # 输出最优的模型参数
                    print(clf.best_params_)
                    print("Grid scores on development set:")
                    for params, mean_score, scores in clf.grid_scores_:
                        print("%0.3f (+/-%0.03f) for %r"
                              % (mean_score, scores.std() * 2, params))
                    print("Detailed classification report:")
                    print("The model is trained on the full development set.")
                    print "模型在测试集上的表现..."
                    # 在测试集上测试最优的模型的泛化能力.
                    y_true, y_pred = y_test, clf.predict(x_test)
                    print classification_report(y_true, y_pred)
