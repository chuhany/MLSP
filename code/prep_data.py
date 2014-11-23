import scipy.io as sio
import numpy
import cPickle
import gzip

train = sio.loadmat('train_32x32.mat')
test = sio.loadmat('test_32x32.mat')

valid_size = 10000
train_size = len(train['y']) - 10000
test_size = len(test['y'])

train_set = (numpy.zeros((train_size, 32 * 32), dtype=numpy.float32), numpy.zeros((train_size), dtype=numpy.int64))
valid_set = (numpy.zeros((valid_size, 32 * 32), dtype=numpy.float32), numpy.zeros((valid_size), dtype=numpy.int64))
test_set = (numpy.zeros((test_size, 32 * 32), dtype=numpy.float32), numpy.zeros((test_size), dtype=numpy.int64))

print "begin..."

for i in range(0, train_size):
    for x in range(0, 32):
        for y in range(0, 32):
            for j in range(0, 3):
                train_set[0][i][32 * x + y] += train['X'][x][y][j][i]
            train_set[0][i][32 * x + y] /= 3
            train_set[0][i][32 * x + y] /= 256
    train_set[1][i] = train['y'][i]

print "train_set finished"

for i in range(0, valid_size):
    for x in range(0, 32):
        for y in range(0, 32):
            for j in range(0, 3):
                valid_set[0][i][32 * x + y] += train['X'][x][y][j][train_size + i]
            valid_set[0][i][32 * x + y] /= 3
            valid_set[0][i][32 * x + y] /= 256
    valid_set[1][i] = train['y'][train_size + i]

print "valid_set finished"

for i in range(0, test_size):
    for x in range(0, 32):
        for y in range(0, 32):
            for j in range(0, 3):
                test_set[0][i][32 * x + y] += test['X'][x][y][j][i]
            test_set[0][i][32 * x + y] /= 3
            test_set[0][i][32 * x + y] /= 256
    test_set[1][i] = test['y'][i]

print "test_set finished"

f = file('svhn.pkl', 'wb')
for  obj in [train_set, valid_set, test_set]:
    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

f_in = open('svhn.pkl', 'rb')
f_out = gzip.open('svhn.pkl.gz', 'wb')
f_out.writelines(f_in)
f_out.close()
f_in.close()