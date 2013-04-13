all: closestPair

closestPair:	closestPair.cu 
		nvcc -lm -O2 closestPair.cu -o closestPair

clean: rm -rf closestPair
