
from collections import defaultdict

import mnist_loader

def main():
    tr_d , va_d , te_d = mnist_loader.load_data()
    #train
    avgs = avg_darkness(tr_d)

    num_correct = sum(int(guess_digit(image,avgs)==digit)
                    for image,digit in zip(te_d[0],te_d[1]))

    print "Baseline classifier using average darkness of image."
    print "%s of %s values correct." % (num_correct, len(te_d[0]))


def avg_darkness(tr_d):
    digits_counts = defaultdict(int)
    darknesses = defaultdict(float)
    for image,digit in zip(tr_d[0],tr_d[1]):
        digits_counts[digit] +=1
        darknesses[digit] += sum(image)
    avgs = defaultdict(float)
    for digit , n in digits_counts.iteritems():
        avgs[digit] = darknesses[digit]/n
    return avgs


def guess_digit(image,avgs):
    darkness = sum(image)
    distances = { k:abs(v-darkness)for k,v in avgs.iteritems()}
    return min(distances,key = distances.get)

if __name__ == "__main__":
    main()
