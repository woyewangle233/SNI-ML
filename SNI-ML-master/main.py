from Muti_MR import *
from config.config import args

path= args.gold_sum_path
PATH = args.sum_path
files=[]
getfile_story(path,files)

print(len(files))


if True:
    count = 0
    for file in files:
        try:
            write2sum(file, write_path=PATH, F=Muti_MR_com,lamada=args.lamada)  # F { Manifold_Rank  Muti_MR_li  Muti_MR_com }
        except ValueError:
            pass


