執行下列指令可Reproduce結果
bash run.sh
最後結果出現在result資料夾中

data資料夾中存放drivendata所提供的資料
model資料夾中存放不同regressor的model
npy資料夾中存放train、test等data的mean、std
result資料夾會存放執行完run.sh後產生的結果
src資料夾中存放train、test等程式碼

xgboost套件的安裝方法與一般套件不同，不是直接使用pip3 install，
所以沒有列入requirements.txt裡，若要安裝，則需使用下列方法：
# git clone --recursive https://github.com/dmlc/xgboost
# cd xgboost
# make
# cd python-package/
# python setup.py install -user
