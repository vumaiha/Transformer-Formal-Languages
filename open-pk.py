import pickle
from src import *

info = pickle.load(open("data/SL_2_3_1_u-bbb/train_corpus.pk","rb"))
print(info.target)