from __future__ import print_function
import six
import numpy as np

import chainer
import chainer.functions as F
import chainer.Links as L

class VIN(chainer.Chain):
    def __init__(self, k=10, l_h=50, l_q=10, l_a=8):
        super(VIN, self).__init__(
                conv1 = L.Convolution2D(2, l_h, 3, stride=1, pad=1),
                conv2 = L.Convolution2D(l_h, 1, 1, stride=1, pad=0, nobias=True),
                conv3 = L.Convolution2D(1, l_q, 3, stride=1, pad=1, nobias=True),
                conv3b = L.Convolution2D(1, l_q, 3, stride=1, pad=1, nobias=True),

                l3 = L.Linear(l_q, l_a, nobias=True),
        )
        self.k = k
        self.train = True

    def __call__(self, x, s1, s2):
        h = F.relu(self.conv(x))
        self.r = self.conv2(h)

        q = self.conv3(self.r)
        self.v = F.max(q, axis=1, keepdims=True)

        for i in xrange(self.k - 1):
            q = self.conv3(self.r) + self.conv3b(self.v)
            self.v = F.max(q, asix=1, keepdims=True)

        q = self.conv3(self.r) + self.conv3b(self.v)
