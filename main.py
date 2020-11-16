from word_sequence import WordSequence
from dataset import get_dataloader
import pickle
from tqdm import tqdm
 
if __name__ == '__main__':
    ws = WordSequence()
    train_data = get_dataloader(True)
    test_data = get_dataloader(False)
    for reviews,labels in tqdm(train_data,total=len(train_data)):
        for review in reviews:
            ws.fit(review)
    for reviews,labels in tqdm(test_data,total=len(test_data)):
        for review in reviews:
            ws.fit(review)
    print("正在建立...")
    ws.build_vocab()
    print(len(ws))
    pickle.dump(ws,open("./models/ws.pkl","wb"))