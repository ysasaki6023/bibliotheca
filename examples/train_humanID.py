# -*- coding: utf-8 -*-
import os,sys,argparse,glob,random,cv2
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.functions.loss import softmax_cross_entropy
from chainer.functions.evaluation import accuracy
from chainer import reporter

# Network definition
class net(chainer.Chain):
    def __init__(self):
        super(net, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.conv0=L.Convolution2D(None, 32, ksize=3, stride=1, pad=0)
            self.conv1=L.Convolution2D(None, 32, ksize=3, stride=1, pad=0)
            self.conv2=L.Convolution2D(None, 32, ksize=3, stride=1, pad=0)
            self.conv3=L.Convolution2D(None, 32, ksize=3, stride=1, pad=0)

            self.convd=L.Convolution2D(None, 32, ksize=1, stride=1, pad=0)

            self.fc1 = L.Linear(None, 4096)
            self.fc2 = L.Linear(None, 4096)
            self.fc3 = L.Linear(None, 2)

    def __call__(self, x1, x2):
        def stream(x):
            h = x
            h = F.relu(self.conv0(h))
            h = F.max_pooling_2d(h,ksize=2,stride=2,pad=0)

            h = F.relu(self.conv1(h))
            h = F.max_pooling_2d(h,ksize=2,stride=2,pad=0)

            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))

            return h

        h1 = stream(x1)
        h2 = stream(x2)

        h  = h1-h2
        h  = F.relu(self.convd(h))
        h  = F.max_pooling_2d(h,ksize=1,stride=1,pad=0)

        #h  = F.flatten(h)

        h  = F.relu(self.fc1(h))
        h  = F.relu(self.fc2(h))
        h  = self.fc3(h)
        h  = F.softmax(h)

        return h

class Model(L.Classifier):
    def __call__(self,x1,x2,t):
        x1 = chainer.Variable(x1)
        x2 = chainer.Variable(x2)
        t  = chainer.Variable(t)
        y = self.predictor(x1,x2)
        loss = F.softmax_cross_entropy(y,t)
        acc = accuracy.accuracy(y,t)
        reporter.report({"accuracy": acc,"loss":loss}, self)
        return loss

class pairIterator(chainer.dataset.iterator.Iterator):
    def __init__(self,nBatch=128,mode="train",samePairFrac=0.5):
        self.sizeX, self.sizeY, self.nColor = 60,160,3
        self.samePairFrac = samePairFrac
        self.nBatch = nBatch
        self.fileList = {}
        for f in glob.glob("HumanID/data/*.png"):
            pid, fid = os.path.basename(f).replace(".png","").split("_")
            pid, fid = int(pid), int(fid)
            if pid%10==0: mode="test"
            else:         mode="train"
            if not pid in self.fileList: self.fileList[pid]=[]
            self.fileList[pid].append(f)
        print("mode={mode} : {fileList}".format(mode=mode,fileList=self.fileList))
        self.nCalls = 0
        return

    @property
    def epoch_detail(self):
        return float(self.nCalls)/len(self.fileList)

    def __next__(self):

        batchList = []
        for _ in range(self.nBatch):
            self.nCalls += 1

            isSame = random.choice([True,False])
            t = np.array(1 if isSame else 0)

            fileList = self.fileList
            pid1 = random.choice(list(fileList.keys()))
            if isSame: pid2 = pid1
            else:
                while True:
                    pid2 = random.choice(list(fileList.keys()))
                    if not pid2==pid1: break

            fname1 = random.choice(fileList[pid1])
            fname2 = random.choice(fileList[pid2])

            img1 = cv2.imread(fname1)
            img2 = cv2.imread(fname2)
            
            img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

            img1 = cv2.resize(img1,(self.sizeX,self.sizeY),interpolation=cv2.INTER_CUBIC)
            img2 = cv2.resize(img2,(self.sizeX,self.sizeY),interpolation=cv2.INTER_CUBIC)

            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)

            # pre-process
            def convImg(x):
                x = np.transpose(x,(2,0,1)) # color, height, width
                x /= 255.
                return x

            img1 = convImg(img1)
            img2 = convImg(img2)

            batchList.append( {"x1":img1,"x2":img2, "t":t} )
        return batchList

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',help='Disable PlotReport extension')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    predictor = net()
    #model = L.Classifier(predictor,lossfun=)
    model = Model(predictor)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    #train, test = chainer.datasets.get_mnist()

    #train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    #test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
    #                                             repeat=False, shuffle=False)
    train_iter = pairIterator(mode="train")
    test_iter  = pairIterator(mode="test")

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__=="__main__":
    main()
