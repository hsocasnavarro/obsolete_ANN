# This is the Neural Network module for python that implements the
# algorithms developed by Hector Socas-Navarro (References needed here...)
#
# Requirements:
#   Uses module Numeric. While this is a non-core module it usually comes
#  included with most python distributions. If not, you should be able to
#  get Numpy that also includes it.
#

class ANN_class:
   """The neural network class

   This class is essentially a structure with the following fields:
   nlayers: integer
   nmaxperlayer: integer
   ninputs: integer
   noutputs: integer
   Nperlayer: integer array with nlayers+1 elements
   Nonlin: integer array with nlayers elements
   W: Float array [nlayers,nmaxperlayer,nmaxperlayer]
   Beta: Float array [nlayers,nmaxperlayer]
   xnorm, xmean: Float arrays of size ninputs
   ynorm, ymean: Float arrays of size noutputs

   HSN 2008

   """

   from Numeric import zeros

   nlayers=nmaxperlayer=ninputs=noutputs=0
   Nperlayer=Nonlin=zeros(1,'i',savespace=1)
   xnorm=xcont=ynorm=ycont=zeros(1,'f',savespace=1)
   W=zeros((1,1,1),'f',savespace=1)
   Beta=zeros((1,1),'f',savespace=1)

def write_ANN(ann, filename):
   from pickle import dump
   
   f=open(filename,'w')
   dump(ann, f)
   f.close()

def read_ANN(filename):
   from pickle import load
   
   f=open(filename,'r')
   ann=load(f)
   f.close()

   return ann

def ANN_forward(ann, Input):
   """Forward propagation of inputs through a neural network.

   This function takes a neural network object and an input vector.
   The inputs are propagated through the network. The output vector
   is obtained upon return

   HSN 2008

   """

# Note: Uses Numeric.py for array manipulation

# This routine propagates forward the inputs through a rectangular Artificial
# Neural Network, and returns the result at the output nodes.
# W(l, i, j) represents the synaptic strength from neuron j in the
# layer l-1 to neuron i in layer l. Beta(l, i) is the bias added to the signal
# in neuron i, layer l. nonlin(l) is an integer vector whose elements are 0 
# if l is a linear layer, or 1 if l is a non-linear layer (tanh is then used
# as activation function). input and output are the inputs and outputs vectors,
# of size ninputs and noutputs, respectively. nperlayer is the number of
# neurons per layer and nlayers is the number of layers (note: the input
# is _NOT_ considered a layer). If the keyword Y is present, the
# neuron values are returned.
#
# Note that nmaxperlayer _MUST BE_ larger than both ninputs and noutputs!!

# Note: Unlike the F90 routine, this one does the preprocessing and
#  postprocessing on-board

# Note on handling indeces, as compared to the F90 routine:
#  Python's array indeces start at 0 whereas F90 normally start at 1 so all
#  array indeces are translated from i to i-1
#  Exceptions are y (first index) and nperlayer, which in F90 start at 0.
#  For these two, i is translated to i
#  Index loops that are iterated from 1 to n in F90 will be iterated here
#  between 0 and n-1. This means that arrays indexed by a variable will look
#  the same in Python and F90, except for y and nperlayer.
#  Examples:
#   F90: W(i,:,:) with i from 1 to n ->Py: W[i][:][:] with i from 0 to n-1
#   F90: y(i,:) with i from 1 to n ->Py: y(i+1,:) with i from 0 to n-1
#   F90: y(i,:) with i from 0 to n ->Py: y(i,:) with i from 0 to n
#  Also note that, because of how Python handles array slices, the
#  equivalent of F90: c(1:2) would be c[0:2] and NOT c[0:1]

   from Numeric import array, zeros, tanh
                
# Parameters
   a, b, bovera, asq=1.7159, 0.666666, 0.388523, 2.94431
# Deconstruct ann
   W=ann.W
   Beta=ann.Beta
   nlayers=ann.nlayers
   nmaxperlayer=ann.nmaxperlayer
   ninputs=ann.ninputs
   noutputs=ann.noutputs
   nperlayer=ann.Nperlayer
   Nonlin=ann.Nonlin
   xmean=ann.xmean
   xnorm=ann.xnorm
   ymean=ann.ymean
   ynorm=ann.ynorm
# Preprocess inputs
   Input=array(Input,'f',savespace=1)
   Input=(Input-xmean)/xnorm
# y and output an output arguments
   y=zeros((nlayers+1,nmaxperlayer),'f',savespace=1)
   y.savespace(1) # This avoids the "Array can not be safely cast..." error
   Output=zeros((noutputs),'f',savespace=1)
# Set input values. Not that y and nperlayer have the first index shifted up 
#     by 1 with respect to the f90 routine
   y[0,0:ninputs]=Input[0:ninputs]
# Propagate forward
   for l in range(nlayers):
      for j in range(nperlayer[l+1]):
         y[l+1, j]=0.
         for k in range(nperlayer[l]):
            y[l+1, j]=y[l+1, j]+W[l, j, k]*y[l, k]
            
         y[l+1,j]=y[l+1,j]+Beta[l,j]

      if (Nonlin[l] != 0): # It's a non-linear layer            
         y[l+1,0:nperlayer[l+1]]=a*tanh(b*y[l+1, 0:nperlayer[l+1]])
# Network output
   Output[0:noutputs]=y[nlayers, 0:noutputs]
# Postprocess output
   Output=Output*ynorm+ymean

   return Output


