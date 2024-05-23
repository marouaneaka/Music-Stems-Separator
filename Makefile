make train:
	python3 src/train.py --pre False --model tcf
make train-pre:
	python3 src/train.py --pre True --model tcf
make predict:
	python3 src/predict.py --model tcfScratch
make predictBest:
	python3 src/predict.py --model tcfBest

make clean:
	rm -rf tmp/*
	rm -f src/*.pyc
	rm -rf output/*