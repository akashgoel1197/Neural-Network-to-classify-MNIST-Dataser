
import mnist_loader

from sklearn import svm

def svm_baseline():
    tr, va, te = mnist_loader.load_data()
    print "loading"
    print "please wait..."
    clf = svm.SVC()
    clf.fit(tr[0],tr[1])
    print "fitting"
    pred = [int(a) for a in clf.predict(te[0])]
    num_correct = sum(int(a==y) for a ,y in zip(pred,te[1]))
    print "Baseline classofer using a svm."
    print "%s of %s values correct." %(num_correct,len(te[0]))

if __name__ == "__main__":
    svm_baseline()
                      
           
