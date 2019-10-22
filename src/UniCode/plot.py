import numpy as np
import matplotlib.pyplot as plt
import sys


def readFile(filename):
    f = open(filename)
    sizes = [40]
    times = [0.0]
    title = ''
    try:
        title = f.readline()
        # skip 1 line
        f.readline()
        while True:
            line = f.readline()
            if line:
                slices = line.split(" ")
                if len(slices) <= 2:
                    break;
                size = int(slices[0])
                time = float(slices[1])
                sizes.append(size)
                times.append(time)
    finally:
        f.close()
    return title, sizes, times

if __name__ == '__main__':

    # old = 'MyMM1'
    # new = 'MyMM2'

    old = sys.argv[1]
    new = sys.argv[2]



    plt.xlabel('size')
    plt.ylabel('gflops')
    t1, x1, y1 = readFile('./figs/output_'+old+'.m')
    plt.plot(x1, y1, label=t1, marker='*')
    t2, x2, y2 = readFile('./figs/output_'+new+'.m')
    plt.plot(x2, y2, label=t2, marker='^')
    plt.legend((old, new), loc='upper right')
    plt.show()