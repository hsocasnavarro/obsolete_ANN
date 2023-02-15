! This routine uses back-propagation to train an ANN with a sample of ndata 
! training vectors xdata, ydata of size ninputs and noutputs, respectively. 
! yerror contains an estimate of the errors in ydata (used for weighting).
! W(l, i, j) represents the synaptic strength from neuron j in the
! layer l-1 to neuron i in layer l. Beta(l, i) is the bias added to the signal
! in neuron i, layer l. nonlin(l) is an integer vector whose elements are 0 
! if l is a linear layer, or 1 if l is a non-linear layer (tanh is then used
! as activation function). nmaxperlayer is the number of
! neurons per layer and nlayers is the number of layers (note: the input
! is _NOT_ considered a layer). If the logical keyword PRINTOUT is present, 
! a printout of the training process is displayed at each iteration. MAXITERS
! is the maximum allowed number of iterations. MU is the global learning 
! rate. The algorithm quits when either maxiters iterations have been
! carried out, when the average error over the whole sample reaches 1.e-3, or
! when it has not decreased in the last 6 iterations. This backpropagation 
! algorithm considers momentum (with a constant of 0.3*Mu) and local learning
! rates.
!
! Note that nmaxperlayer _MUST BE_ .ge. than both ninputs and noutputs!!
!
Subroutine Train_ANN(W, Beta, Nonlin, nperlayer, xdata, ydata, yerror, &
     maxndata, ndata, nlayers, nmaxperlayer, ninputs, noutputs, maxninputs, &
     maxnoutputs, printout, maxiters, mu)
Implicit None
Real, Parameter :: a=1.7159, b=0.666666, bovera=0.388523, asq=2.94431
!Real, Parameter :: a=1., b=1., bovera=1., asq=1.
Integer, Parameter :: nstored=200
Integer :: nlayers, nmaxperlayer, ninputs, noutputs, ndata, i, j, k, l, niters, &
     maxiters, maxninputs, maxnoutputs, maxndata
Logical :: printout, done, Diverges
Real :: averr, alpha, mu
Real, Dimension(nstored) :: stored_err
Integer, Dimension(nlayers) :: Nonlin
Integer, Dimension(0:nlayers) :: nperlayer
Real, Dimension(nlayers, nmaxperlayer, nmaxperlayer) :: W, dW, mul, &
     OldsignW, SamesignW, ChangesignW, Winit, OlddW
Real, Dimension(0:nlayers, nmaxperlayer) :: y
Real, Dimension(nmaxperlayer) :: fprime
Real, Dimension(nlayers, nmaxperlayer) :: Beta, Binit, delta, dB, OlddB, mulb, &
     OldsignB, SamesignB, ChangesignB
Real, Dimension(ninputs) :: Inputs
Real, Dimension(noutputs) :: Outputs
Real, Dimension(maxndata, maxninputs) :: xdata
Real, Dimension(maxndata, maxnoutputs) :: ydata, yerror
!
! Set Winit and Binit to be able of going back if the algorithm diverges.
!
Winit=W
Binit=Beta
!
! Some initializations
!
OldsignW=0.
OldsignB=0.
done=.FALSE.
niters=0
stored_err(1:nstored)=1.e10
Mu=Mu/ndata
alpha=0.9
dW=0.
dB=0.
!
! Set weights and biases to zero for non-used neurons
!
Do i=1, nlayers
   Do j=nperlayer(i)+1, nmaxperlayer
      W(i, j, 1:nmaxperlayer)=0.
      Beta(i, j)=0.
   End do
End do
!
! Start iterations
!
If (printout) Open (43,file='iters.log')
Do while (.not. Done .and. niters .lt. maxiters)
   niters=niters+1
   averr=0.
   OlddW=dW
   OlddB=dB
   dW=0.
   dB=0.
   delta=0.
!$ omp parallel
!$ omp do private (inputs,outputs,fprime,delta,y,l,j,k) reduction(+:dB,dW,averr) 
   Do i=1, ndata ! Loop in training points
      inputs(1:ninputs)=xdata(i, 1:ninputs)
!
! Do forward propagation and compute outputs
!
      Call ANN_forward(W, Beta, Nonlin, inputs, outputs, nlayers, &
           nmaxperlayer, nperlayer, ninputs, noutputs, y)
!
! Error in output layer
!
      If (Nonlin(nlayers) .eq. 0) then ! Output layer is linear
         fprime(1:noutputs)=1.
      Else ! It's non-linear (tanh)
         fprime(1:noutputs)=bovera*(asq - &
              y(nlayers, 1:noutputs)*y(nlayers, 1:noutputs))
      End if
      delta(nlayers, 1:noutputs)=(ydata(i, 1:noutputs) - &
           outputs(1:noutputs))*fprime(1:noutputs)
!
! Define tolerance tube
!
      Do l=1, noutputs ! Error=0 if smaller than yerror
         If (abs(ydata(i, l)-outputs(l)) .lt. yerror(i, l)) &
              delta(nlayers, l)=0.
      End do
!
! Update average error and check for divergence
!
      averr=averr+Sum(Abs(ydata(i,1:noutputs)-outputs(1:noutputs)))/ &
           ndata/noutputs
      If (averr .gt. 10.) & ! Divergence. Restart with smaller mu
           Diverges=.TRUE.
!
! Back-propagation of the error signal
!
      Do l=nlayers-1, 1, -1
         If (Nonlin(l) .eq. 0) then ! Linear layer
            fprime(1:nperlayer(l))=1.
         Else ! It's non-linear (tanh)
            fprime(1:nperlayer(l))=bovera*(asq - &
                 y(l, 1:nperlayer(l))*y(l, 1:nperlayer(l)))
         End if
!         delta(l, 1:nperlayer(l))=Matmul(Transpose(W(l+1,1:nperlayer(l),1:nperlayer(l+1))), delta(l+1,1:nperlayer(l+1)))
         Do j=1, nperlayer(l)
            delta(l, j)=0.
            Do k=1, nperlayer(l+1)
               delta(l, j)=delta(l, j)+W(l+1, j, k)*delta(l+1, k)
            End do
         End do
!            
         delta(l, 1:nperlayer(l))=delta(l, 1:nperlayer(l))*fprime(1:nperlayer(l))
      End do
!
! Put contributions in dB and dW to update Beta and W later.
!
      dB=dB+delta
      Do l=1, nlayers
         Do j=1, nperlayer(l)
            Do k=1, nperlayer(l-1)
               dW(l, j, k)=dW(l, j, k)+delta(l, j)*y(l-1, k)
            End do
         End do
      End do
   End do ! Next training point

!$ omp end do
!$ omp end parallel
!
! Is the iteration divergent?
!
   If (Diverges) then ! If so, reset everything and decrease mu
      W=Winit
      Beta=Binit
      dW=0.
      dB=0.
      OlddW=0.
      OlddB=0.
      Mu=Mu/5.
      stored_err=1.e10
      Diverges=.FALSE.
   End if
!
! Manage local learning rates based on the behavior of local gradient signs
!
   mul=1.
   mulB=1.
   Do l=1, nlayers
      Do j=1, nperlayer(l)
         Do k=1, nperlayer(l)
            If (Sign(1., dW(l, j, k)) .eq. OldsignW(l, j, k)) then
               ChangesignW(l, j, k)=0.
               SamesignW(l, j, k)=SamesignW(l, j, k)+1
               If (SamesignW(l, j, k) .ge. 6) &
                    mul(l, j, k)=SamesignW(l, j, k)!**2
            Else
               SamesignW(l, j, k)=0
               ChangesignW(l, j, k)=ChangesignW(l, j, k)+1
               If (ChangesignW(l, j, k) .ge. 3) &
                    mul(l, j, k)=1./(ChangesignW(l, j, k))!**2)
            End if
            OldsignW(l, j, k)=Sign(1., dW(l, j, k))
         End do
      End do
   End do
   Do l=1, nlayers
      Do j=1, nperlayer(l)
         changesignb(l,j)=0.
         If (Sign(1., dB(l, j)) .eq. OldsignB(l, j)) then
            ChangesignB(l, j)=0.
            SamesignB(l, j)=SamesignB(l, j)+1
            If (SamesignB(l, j) .ge. 6) &
                 mulB(l, j)=SamesignB(l, j)!**2
         Else
            SamesignB(l, j)=0
            ChangesignB(l, j)=ChangesignB(l, j)+1
            If (ChangesignB(l, j) .ge. 3) &
                 mulB(l, j)=1./(ChangesignB(l, j))!**2)
         End if
      End do
   End do
!
! Batch training
!
   dW=dW+alpha*olddW
   dB=dB+alpha*olddB

   W = W +  Mu*Mul*(dW) ! Update synaptic strenghts
   Beta = Beta + Mu*Mulb*(dB) ! Update biases
!   W = W +  Mu*Mul*(alpha*olddW + dW) ! Update synaptic strenghts
!   Beta = Beta + Mu*Mulb*(alpha*olddB + dB) ! Update biases
!
! Printout
!
   If (printout) then
      Print *,'Iter=',niters,' Av error=',averr,' Mu=',Mu
      Write (43,*) 'Iter=',niters,' Av error=',averr,' Mu=',Mu
   End if
!
! Done?
! 
!   If (averr .le. 2.5e-2) Done=.TRUE.
   stored_err(1:nstored-1)=stored_err(2:nstored)
   stored_err(nstored)=averr
   If (Minval(stored_err(1:nstored/2)) .le. &
        Minval(stored_err(nstored/2+1:nstored))) Done=.TRUE.
!

   If (printout .and. niters/10 .eq. niters/10.) then
      Open (42, file='tmpnet.dat')
      Do i=1, nlayers
         Do j=1, nmaxperlayer
            Do k=1, nmaxperlayer
               Write (42,*) W(i,j,k)
            End do
         End do
      End do
      Do i=1, nlayers
         Do j=1, nmaxperlayer
            Write (42,*) Beta(i,j)
         End do
      End do
      Close (42)
   End if
End do ! Loop in iterations
Mu=Mu*ndata
!
! Done!
!
If (printout) Close (43)
Return
!
End Subroutine Train_ANN

