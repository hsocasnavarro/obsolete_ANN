Program ANN
  Implicit None
  Integer, Parameter :: nlayers=2, nmaxperlayer=27, noutputs=1, &
       maxndata=10000, ntraining=maxndata, ninputs=nmaxperlayer
  Integer :: nactual, i, j, k, l, ipoint, status, unit
  Integer :: read_network
  Integer, Dimension(nlayers) :: Nonlin
  Integer, Dimension(0:nlayers) :: nperlayer
  Real, Dimension(nlayers, nmaxperlayer, nmaxperlayer) :: W
  Real, Dimension(nlayers+1, nmaxperlayer, nmaxperlayer) :: y
  Real, Dimension(nlayers, nmaxperlayer) :: Beta
  Real, Dimension(ninputs) :: xdata, xdatap
  Real, Dimension(ninputs) :: xnorm, xmean, inputs
  Real, Dimension(noutputs) :: ynorm, ymean, outputs
  Real, Dimension(maxndata) :: home, away, date, resul
  Real :: kk, averr, Mu, err
  Integer :: niters, ndata, maxteam
!
  Open (32, file='inputs.dat',form='unformatted')
  Read (32) inputs(1:27)
  Close (32)
!
! Construct ANN
!
  Nonlin(1:nlayers)=1
  nperlayer(0)=ninputs
  nperlayer(1)=2
  nperlayer(2)=1
!
! Read previously saved ANN
!
  Open (31, file='Invert.net',Form='unformatted')
  Read (31) W(1:nlayers, 1:nmaxperlayer, 1:nmaxperlayer)
  Read (31) Beta(1:nlayers, 1:nmaxperlayer)
  Read (31) xnorm(1:ninputs), xmean(1:ninputs)
  Read (31) ynorm(1:noutputs), ymean(1:noutputs)
  Close (31)     
!
  Print *,'Preconditioning inputs ...'
!
! Preprocess inputs and outputs
!
  Do i=1, ninputs
     xdatap(i)=(inputs(i)-xmean(i))/xnorm(i)
  End do
!
  inputs=xdatap
  Call ANN_forward(W, Beta, Nonlin, inputs, outputs, nlayers, &
       nmaxperlayer, nperlayer, ninputs, noutputs, y)
  Print *, &
          outputs(1:noutputs)*ynorm(1:noutputs)+ymean(1:noutputs)
!
End Program ANN
