Program train
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
  Real, Dimension(maxndata, ninputs) :: xdata, xdatap
  Real, Dimension(maxndata, noutputs) :: ydata, ydatap, yerror
  Real, Dimension(ninputs) :: xnorm, xmean, inputs
  Real, Dimension(noutputs) :: ynorm, ymean, outputs
  Real, Dimension(28,maxndata) :: dbase
  Real :: kk, averr, Mu, err
  Integer :: niters, ndata, maxteam
!
! Set parameters
!
  Mu=0.0005
  read_network=.FALSE.
  niters=10000
!
  Open (40,file='logfile')
  Close (40)
  Do while (.True.) ! Endless loop
  Open (40,file='logfile',Position='APPEND')
!
  Open (32, file='dbase.dat',form='unformatted')
  Read (32) nactual
  If (nactual .gt. maxndata) then
     Print *,'maxndata=',maxndata,', nactual=',nactual
     Stop
  End if
  ndata=nactual
  print *,'Ndata=',ndata
  Do i=1, 28
     Read (32) dbase(i,1:ndata)
  End do
  Close (32)
  Do ipoint=1, 27
     xdata(1:ndata,ipoint)=dbase(ipoint,1:ndata)
  End do
  ydata(1:ndata,1)=dbase(28,1:ndata)
!
! Construct ANN
!
  Nonlin(1:nlayers)=1
  nperlayer(0)=ninputs
  nperlayer(1)=2
  nperlayer(2)=1
!
! Random initialization
!
  Do l=1, nlayers
     Do j=1, nmaxperlayer
        Do k=1, nmaxperlayer
           Call Random_number(kk)
           kk=kk-.5
           W(l, j, k)=kk/Sqrt(nperlayer(l)*1.)
        End do
     End do
  End do
!
  Do l=1, nlayers
     Do j=1, nmaxperlayer
        Call Random_number(kk)
        kk=kk-.5
        Beta(l, j)=kk*.1
     End do
  End do
!
! Initialize with previously saved ANN
!
  If (read_network) then
     Open (31, file='Invert.net',Form='unformatted')
     Read (31) W(1:nlayers, 1:nmaxperlayer, 1:nmaxperlayer)
     Read (31) Beta(1:nlayers, 1:nmaxperlayer)
     Read (31) xnorm(1:ninputs), xmean(1:ninputs)
     Read (31) ynorm(1:noutputs), ymean(1:noutputs)
     Close (31)     
!     Open (31, file='tmpnet.dat')
!     Do i=1, nlayers
!        Do j=1, nmaxperlayer
!           Do k=1, nmaxperlayer
!              Read (31,*) W(i,j,k)
!           End do
!        End do
!     End do
!     Do i=1, nlayers
!        Do j=1, nmaxperlayer
!           Read (31,*) Beta(i,j)
!        End do
!     End do
  End if
!
  Print *,'Preconditioning inputs ...'
  If (.not. Read_network) then
     Print *,'Calculating normalizations'
     Do i=1, noutputs
        ynorm(i)=(Maxval(ydata(1:ndata,i))-Minval(ydata(1:ndata,i)))/2.
        ymean(i)=Sum(ydata(1:ndata,i))/ndata
!           ynorm(i)=Sqrt(Sum((ydata(1:ndata,i)-ymean(i))**2)/(ndata-1))
     End do
     Do i=1, ninputs
        xnorm(i)=(Maxval(xdata(1:ndata,i))-Minval(xdata(1:ndata,i)))/2.
        xmean(i)=Sum(xdata(1:ndata,i))/ndata
!           xnorm(i)=Sqrt(Sum((xdata(1:ndata,i)-xmean(i))**2)/(ndata-1))
!           print *,'xm=',xmean(i),xnorm(i)
     End do
     Where (xnorm .eq. 0) xnorm=1.
     Where (ynorm .eq. 0) ynorm=1.
  End if
  Read_network=.TRUE.
!
! Start training
!
  Print *,'Ok. Starting training'
!
! Preprocess inputs and outputs
!
  Do i=1, ninputs
     xdatap(1:ndata,i)=(xdata(1:ndata,i)-xmean(i))/xnorm(i)
  End do
  Do i=1, noutputs
     ydatap(1:ndata,i)=(ydata(1:ndata,i)-ymean(i))/ynorm(i)
     yerror(1:ndata,i)=0.
  End do
!
  Open (31, file='Invert.net', Form='unformatted')
  Write (31) W(1:nlayers, 1:nmaxperlayer, 1:nmaxperlayer)
  Write (31) Beta(1:nlayers, 1:nmaxperlayer)
  Write (31) xnorm(1:ninputs), xmean(1:ninputs)
  Write (31) ynorm(1:noutputs), ymean(1:noutputs)
  Close (31)  
!
! Train
!
  Call Train_ANN(W, Beta, Nonlin, nperlayer, xdatap, ydatap, yerror, &
       maxndata, ndata, nlayers, nmaxperlayer, &
       ninputs, 1, ninputs, noutputs, .TRUE., niters, Mu)
!
  Open (31, file='Invert.net', Form='unformatted')
  Write (31) W(1:nlayers, 1:nmaxperlayer, 1:nmaxperlayer)
  Write (31) Beta(1:nlayers, 1:nmaxperlayer)
  Write (31) xnorm(1:ninputs), xmean(1:ninputs)
  Write (31) ynorm(1:noutputs), ymean(1:noutputs)
  Close (31)  
!
! Ok. Done
!
! Test against other set
!
  Open (32, file='dbase.dat',form='unformatted')
  Read (32) nactual
  Read (32) dbase(i,1:nactual)
  Close (32)
  Do ipoint=1, 27
     xdata(1:nactual,ipoint)=dbase(ipoint,1:nactual)
  End do
  ydata(1:nactual,1)=dbase(28,1:nactual)
  Do i=1, ninputs
     xdatap(1:nactual,i)=(xdata(1:nactual,i)-xmean(i))/xnorm(i)
  End do
  Do i=1, noutputs
     ydatap(1:nactual,i)=(ydata(1:nactual,i)-ymean(i))/ynorm(i)
  End do
  Open (32, file='test.dat')
  averr=0.
  Do i=1, nactual
     inputs=xdatap(i,:)
     Call ANN_forward(W, Beta, Nonlin, inputs, outputs, nlayers, &
          nmaxperlayer, nperlayer, ninputs, noutputs, y)
     Print *,'Exact value: ',ydatap(i, 1:noutputs)*ynorm(1:noutputs)+ymean(1:noutputs)
     Print *,'ANN value  : ',outputs(1:noutputs)*ynorm(1:noutputs)+ymean(1:noutputs)
     Print *,'*******************************************'
     Write (32,*) &
          ydatap(i, 1:noutputs)*ynorm(1:noutputs)+ymean(1:noutputs), &
          outputs(1:noutputs)*ynorm(1:noutputs)+ymean(1:noutputs)
     err=Sum(Abs(outputs(1:1)-ydatap(i, 1:1)))/1.
     averr=averr+err
  End do
  averr=averr/nactual
  Print *,'Average error in the validation set: ',averr
  Write (40,*) 'Averr=',averr
  Close (32)
  Close (40)
!
  End do ! Endless loop
!
End Program train
