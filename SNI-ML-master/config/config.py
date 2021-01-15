import argparse
parser = argparse.ArgumentParser()


parser.add_argument("-gold_sum_path", default='./data/gold-sum', type=str,)
parser.add_argument("-sum_path", default='./data/SNI-ML-sum/', type=str, )
parser.add_argument("-re_path", default='./data/re_s/', type=str, )
parser.add_argument("-words", default='./data/filed_words/', type=str, )
parser.add_argument("-sn", default='./data/SN/', type=str, )
parser.add_argument("-vector", default='../GoogleNews-vectors-negative300.bin', type=str, )



parser.add_argument("-lamada", default=0.8, type=float, )
parser.add_argument("-mu", default=0.2, type=float, )
parser.add_argument("-eta", default=0.3, type=float, )
parser.add_argument("-lam", default=0.9, type=float, )

args = parser.parse_args()

print(args)