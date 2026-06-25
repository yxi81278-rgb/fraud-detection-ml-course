# -*- coding: utf-8 -*-
# 04_rewrite.py — Fraud-R1终极版: 句子删除 + 强同义词替换 + 社会工程诱导

import os, sys, re, random
import numpy as np
import jieba

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (PROJECT_ROOT, DATA_PROCESSED, DATA_ATTACKED,
                   clean_text, save_pickle, load_pickle)

# ===== 强同义词替换表 =====
SYNONYM_MAP = {
    '客服': ['服务人员', '工作人员', '接线员', '专员'],
    '退款': ['返款', '退还', '退费', '退回款项'],
    '投资': ['理财', '资产配置', '资金管理', '财富规划'],
    '账户': ['户头', '账号', '名下账户', '登记信息'],
    '公安机关': ['相关部门', '执法单位', '管理机关', '主管单位'],
    '下载': ['获取', '安装', '装入', '接收'],
    '链接': ['网址', '地址', '入口', '通道'],
    '验证码': ['确认码', '校验码', '识别码', '核验码'],
    '转账': ['汇款', '划款', '转款', '资金转移'],
    '冻结': ['限制', '管控', '监管', '暂时保管'],
    '涉嫌': ['涉及', '牵涉', '关联', '可能存在'],
    '犯罪': ['违规', '不当行为', '异常情况', '问题'],
    '安全': ['保障', '防护', '保护', '安稳'],
    '优惠': ['福利', '补贴', '让利', '特惠'],
    '紧急': ['重要', '加急', '优先', '特殊'],
    '资金': ['款项', '资产', '钱款', '金额'],
    '银行卡': ['储蓄卡', '金融卡', '卡片', '卡'],
    '身份证': ['证件', '身份证明', '个人证件'],
    '密码': ['口令', '暗码', '密令'],
    '系统': ['平台', '后台', '程序', '服务器'],
    '官方': ['正规', '授权', '认证', '指定'],
    '平台': ['渠道', '窗口', '界面', '端口'],
    '诈骗': ['欺骗', '欺瞒', '虚假行为'],
    '违法': ['不合规', '不符合规定', '涉嫌违规'],
    '洗钱': ['资金异常流动', '可疑资金往来'],
    '涉案': ['涉及案件', '关联事务'],
    '征信': ['信用记录', '信誉评估'],
    '扣款': ['划扣', '支出', '交易'],
    '限额': ['上限', '限制额度', '封顶'],
    '风险': ['隐患', '问题', '异常'],
}

# 欺诈指示词列表——这些词出现说明极可能是欺诈
FRAUD_SIGNAL_WORDS = [
    '公安机关', '涉嫌', '犯罪', '冻结', '验证码', '安全账户',
    '洗钱', '涉案', '违法', '通缉', '逮捕', '立案',
    '国家反诈', '征信', '扣款', '限额',
]

# ===== 社会工程话术 =====
STRATEGY_PHRASES = {
    'trust': {
        'opening': ['您好，我们是正规认证机构，请放心。'],
        'mid': ['我们已服务百万用户，安全有保障。', '您的信息绝不会泄露。'],
    },
    'urgency': {
        'opening': ['您好，这里是紧急通知。'],
        'mid': ['仅剩最后3个名额，限时24小时。', '系统今晚关闭，请立即处理。'],
    },
    'emotion': {
        'opening': ['您好，此事关系到您的切身利益。'],
        'mid': ['为了家人未来，请务必重视。', '一旦错过无法挽回。'],
    },
    'authority': {
        'opening': ['您好，这里是国家反诈中心联合执法办公室。'],
        'mid': ['根据反诈法规定……', '国家网络安全中心已下发预警。'],
    },
    'comprehensive': {
        'opening': ['您好，我是国家金融监管局专员李华，工号FZ88421。'],
        'mid': ['该风险限时2小时处理，逾期将冻结。', '直接影响您的财产和征信。'],
    },
}


def synonym_replace(text, ratio=0.85):
    """强力同义词替换: 85%的可替换词被替换"""
    words = list(jieba.cut(text))
    new_words = []
    for w in words:
        if w in SYNONYM_MAP and random.random() < ratio:
            new_words.append(random.choice(SYNONYM_MAP[w]))
        else:
            new_words.append(w)
    return ''.join(new_words)


def delete_fraud_lines(text, delete_prob=0.3):
    """删除包含欺诈信号的句子"""
    lines = text.split('\n')
    new_lines = []
    for line in lines:
        if line.strip().startswith('left:'):
            content = line.replace('left:', '').strip()
            has_signal = any(signal in content for signal in FRAUD_SIGNAL_WORDS)
            if has_signal and random.random() < delete_prob:
                continue  # 删除这一行
        new_lines.append(line)
    return '\n'.join(new_lines)


def insert_phrases(text, strategy):
    """插入社会工程话术"""
    # Step 1: 删除欺诈信号句 (30%概率)
    text = delete_fraud_lines(text, delete_prob=0.35)
    
    # Step 2: 强力同义词替换 (85%概率)
    text = synonym_replace(text, ratio=0.85)
    
    # Step 3: 社会工程话术
    phrases = STRATEGY_PHRASES[strategy]
    lines = text.split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        if line.strip().startswith('left:'):
            if i == 0:
                new_lines.append(f'left: {random.choice(phrases["opening"])}')
        new_lines.append(line)
    
    return '\n'.join(new_lines)


def rewrite_with_strategy(texts, strategy):
    rewritten = []
    for text in texts:
        rw = insert_phrases(text, strategy)
        rewritten.append(rw)
    return rewritten


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=os.path.join(DATA_PROCESSED, 'prepared_data.pkl'))
    parser.add_argument('--strategies', type=str, nargs='+',
                        default=['trust', 'urgency', 'emotion', 'authority', 'comprehensive'])
    args = parser.parse_args()

    print('Loading data...')
    data = load_pickle(args.data)
    X_test_fraud_orig = data.get('X_test_fraud_orig', [])
    if not X_test_fraud_orig:
        X_test = data['X_test']; y_test = data['y_test']
        X_test_fraud_orig = [X_test[i] for i, l in enumerate(y_test) if l == 1]
    print(f'Original fraud test samples: {len(X_test_fraud_orig)}')

    print('\nApplying ULTIMATE Fraud-R1: delete signals + synonym replace + social engineering...')
    results = {'original': X_test_fraud_orig}
    for s in args.strategies:
        print(f'  Strategy: {s} ...')
        results[s] = rewrite_with_strategy(X_test_fraud_orig, s)
    print(f'Done.')

    os.makedirs(DATA_ATTACKED, exist_ok=True)
    save_pickle(results, os.path.join(DATA_ATTACKED, 'rewritten_test_samples.pkl'))
    print(f'Saved.')


if __name__ == '__main__':
    main()
