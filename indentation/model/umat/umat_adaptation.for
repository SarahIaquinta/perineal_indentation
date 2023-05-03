C --------------------------------------------------------------------------
C Subrutina UMAT para material hiperelastico anisotropo inelastico (da�o y viscoelasticidad)
C Noviembre 2005 Fany
C  Separamos la contribucion de las dos familas de fibras
C --------------------------------------------------------------------------
C IMPORTANTE: Antes de utilizar las rutinas aqui
C             programadas, compruebe los INCLUDE (especialmente en linux)
C             estableciendo el "path" correspondiente.
C
C             De igual manera, compruebe la existencia y la ubicacion
C             del fichero steps.txt en la subrutina UEXTERNALDB
C			De igual manera, compruebe la existencia del fichero
C			param_umat.txt, actualice el numero de nudos y elementos de usuario
C			y los flag segun use fibras, preten, crecimiento
C --------------------------------------------------------------------------
C
C --------------------------------------------------------------------------
C Subrutina UMAT para material hiperelastico anisotropo inelastico (da�o y viscoelasticidad)
C Noviembre 2005
C GEMM-Unizar
C --------------------------------------------------------------------------
C        1         2         3         4         5         6         7
C234567890123456789012345678901234567890123456789012345678901234567890
C
      SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1 RPL,DDSDDT,DRPLDE,DRPLDT,
     2 STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME,
     3 NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,
     4 CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,KSTEP,KINC)
C
      INCLUDE 'ABA_PARAM.INC'
      INCLUDE 'param_umat.txt'
      COMMON /KIDX/ NEFIB,NPOSFIB,NEF0,NPOSF0
      COMMON /KFIB/ NEIDXF0,NEIDXOR,FIBORI,F0EL
C
      CHARACTER*80 CMNAME
      DIMENSION STRESS(NTENS),STATEV(NSTATV),
     1 DDSDDE(NTENS,NTENS),DDSDDT(NTENS),DRPLDE(NTENS),
     2 STRAN(NTENS),DSTRAN(NTENS),TIME(2),PREDEF(1),DPRED(1),
     3 PROPS(NPROPS),COORDS(3),DROT(3,3),DFGRD0(3,3),DFGRD1(3,3)
      DIMENSION NEIDXF0(KELEM),NEIDXOR(KELEM)
      logical fou
      integer idx
      real*8 FIBORI(KELEM,KNPG,6),F0EL(KELEM,KNPG,9)
      real*8 jac,jisoc
      real*8 dw1(4),dw2(10),wener(4)
      real*8 FG(3,3),F0(3,3),invF0(3,3)
      real*8 tenb(NTENS,1),tenb2(NTENS,1),inva(4),Sa2(NTENS,1)
      real*8 na0(3,1),na(3,1),nb0(3,1),nb(3,1),norma0,normb0
      real*8 pbar,poct,derdamM,damM(6),damF(12),derdamF1, derdamF2
      real*8 Se(NTENS,1),Si(NTENS,1),Sa1(NTENS,1),Sv(NTENS,1)
      real*8 Cv(NTENS,NTENS),Ci(NTENS,NTENS),Ca1(NTENS,NTENS) 
      real*8 Ctot(NTENS,NTENS),Hs(NTENS,NTENS), Ca2(NTENS,NTENS)
	real*8 Hn(NTENS,3),Sn(NTENS,3),aneurg(4)
C
C Initializing
C
       na=0.0; na0=0.0; nb=0.0; nb0=0.0
       damM=0.0; damf=0.0; wener=0.0;
	 derdamM=0.0; derdamF1=0.0;	 derdamF2=0.0;
       damM=STATEV(1:6); damf=STATEV(14:25); 
       Hs=0.0
	 Hn(:,1)=STATEV(26:31); Sn(:,1)=STATEV(32:37);
	 Hn(:,2)=STATEV(38:43); Sn(:,2)=STATEV(44:49);
	 Hn(:,3)=STATEV(50:55); Sn(:,3)=STATEV(56:61);
	 if (abs(PROPS(1)-8.0).LT.1e-6) then
          aneurg(1:4)=STATEV(62:65)
          PROPS(8)=STATEV(66)
          PROPS(10)=STATEV(67)
       endif

       call eye(3,F0)
       call eye(3,invF0)
C
C Recovering inital strain
C
       if (PRETEN_FLAG .GT. 0.0) then
          if (NOEL .ne. NEF0) then
             NEF0=NOEL
             I=1
             do while(NEIDXF0(I) .NE. NOEL)
               I=I+1
             end do
             NPOSF0=I
C             call  b_search (NEIDXOR,KELEM,NOEL,NPOSFIB,fou)
          endif
          F0(:,1)=F0EL(NPOSF0,npt,(/1:3/))
          F0(:,2)=F0EL(NPOSF0,npt,(/4:6/))
          F0(:,3)=F0EL(NPOSF0,npt,(/7:9/))
          call inv(F0,invF0)
       endif

	if (TIME(1).GT. 0.2) then
	  Hs=0.0
	end if
C
C Recovering fiber information
C
	 if (abs(PROPS(49)-0.0).LT.1.0E-6) then
          na0(:,1)=(/0.0,0.0,0.0/)
          nb0(:,1)=(/0.0,0.0,0.0/)
       elseif (abs(PROPS(49)-1.0).LT.1.0E-6) then
          na0(:,1)=PROPS(50:52)
          nb0(:,1)=PROPS(53:55)
	 elseif (abs(PROPS(49)-2.0).LT.1.0E-6) then
	    if ((KSTEP.eq.1).and.(KINC.eq.1).and.(STATEV(13).eq.0)) then 
             call genera_fibras_vena(COORDS,PROPS,NPROPS,na0,nb0)
             STATEV(7:9)=na0(:,1)
             STATEV(10:12)=nb0(:,1) 
             STATEV(13)=1.0
		else
             na0(:,1)=STATEV(7:9) 
             nb0(:,1)=STATEV(10:12)
		end if
	 elseif ((FIB_FICHERO .GT. 0.0).and.(PROPS(49).GE.4.0)) then
          if (NOEL .ne. NEFIB) then
             NEFIB=NOEL
             I=1
             do while(NEIDXOR(I) .NE. NOEL)
                I=I+1
             end do
             NPOSFIB=I
C             call  b_search (NEIDXOR,KELEM,NOEL,NPOSFIB,fou)
          endif
          na0(:,1)=FIBORI(NPOSFIB,npt,(/1:3/))
          nb0(:,1)=FIBORI(NPOSFIB,npt,(/4:6/))
       endif
C
C Calculating the Elasticity Tensor
C
       na0=matmul(invF0,na0)
       norma0=sqrt(dot_product(na0(:,1),na0(:,1)))
       if (norma0 .gt. 1E-6) then
         na0=na0/norma0
       endif
       nb0=matmul(invF0,nb0)
       normb0=sqrt(dot_product(nb0(:,1),nb0(:,1)))
       if (normb0 .gt. 1E-6) then
         nb0=nb0/normb0
       endif
       call det(DFGRD1,jac)
       jisoc=jac**(-1.0/3.0)
       FG=jisoc*matmul(DFGRD1,F0)
       call deftensors(FG,na0,nb0,tenb,tenb2,inva)	      
	 call derW(NPROPS,PROPS,inva,aneurg,wener,dw1,dw2)
C 
C verificamos si existe da�o en matriz y fibras.
C Tenemos tres tipos de da�o:
C	PROPS(17)=1.0 da�o discontinuo de Simo 1987c
C	PROPS(17)=2.0 da�o continuo y discontinuo de Miehe 1997
C	PROPS(17)=3.0 da�o pseudolastico de Ogden & Roxburg 1999
C	PROPS(17)=4.0 da�o viscos de Ju 1999
C	Para el caso del modelo estocastico fibrado (PROPS(1)=4.0) se emplea el da�o discontinuo de Simo 1987c solo para la matriz
C
       if(abs(PROPS(1)-4.0).LT.1e-6) then
        call dam_iso_func(NPROPS,PROPS,inva,damM,derdamM)
       else if(abs(PROPS(17)-1.0).LT.1e-6) then
        call DMG_matriz(PROPS,NPROPS,wener,damM,derdamM)
        call DMG_fibras(PROPS,NPROPS,wener,damF,derdamF1,derdamF2)
	 else if(abs(PROPS(17)-2.0).LT.1e-6) then
        call DMGcont_matriz(PROPS,NPROPS,wener,damM,derdamM)
	  call DMGcont_fibras(PROPS,NPROPS,wener,damF,derdamF1,derdamF2)
	 else if(abs(PROPS(17)-3.0).LT.1e-6) then
        call DMGpseudo_matriz(PROPS,NPROPS,wener,damM,derdamM)
	  call DMGpseudo_fibras(PROPS,NPROPS,wener,damF,derdamF1,derdamF2)
	 else if(abs(PROPS(17)-4.0).LT.1e-6) then
        call DMGvis_matriz(PROPS,NPROPS,dtime,wener,damM,derdamM)
        call DMGvis_fibras(PROPS,NPROPS,dtime,wener,damF,
	1       derdamF1,derdamF2)
	 endif
	 na=matmul(FG,na0)
       if (abs(inva(3))>1.0E-6) then
         na=na/sqrt(inva(3))
       endif
       nb=matmul(FG,nb0)
       if (abs(inva(4))>1E-6) then
         nb=nb/sqrt(inva(4))
       endif
C
C   Stresses
C
       poct=2.0*(jac-1.0)/PROPS(2)
       Se=0.0; Sv=0.0; Si=0.0; Sa=0.0;
       call sigma(ntens,poct,dw1,inva,na,nb,tenb,tenb2,jac,Sv,
     1	    Si,Sa1,Sa2)
	 Si= (1-damM(2))*Si 
	 Sa1= (1.0-damF(2))*Sa1
	 Sa2= (1.0-damF(8))*Sa2
C
C   Elasticity Tensor
C
       pbar=2.0*(2.0*jac-1.0)/PROPS(2)
       Ctot=0.0; Cv=0.0; Ci=0.0; Ca1=0.0; Ca2=0.0;
       call Cvol(ntens,poct,pbar,Cv)
       call Ciso(ntens,dw1,dw2,inva,tenb,tenb2,jac,Ci)
	 Ci= (1.0-damM(2))*Ci
       if(abs(PROPS(1)).GT.0.0) then
         call Caniso2(ntens,dw1,dw2,inva,na,nb,tenb,tenb2,jac,Ca1,Ca2)
         Ca1= (1.0-damF(2)) * Ca1
	   Ca2= (1.0-damF(8))* Ca2
       endif
C
C Actualizamos la viscoelasticidad
C Tenemos tres tipos de viscoelasticidad:
C	PROPS(33)=1.0 visco de Simo 1987c tipo kelvinVoigt
C	PROPS(17)=2.0 visco de Holzapfel 2002 tipo Maxwell
C	PROPS(17)=3.0 visco de Simo 1987c tipo kelvinVoigt nolineal con parametros modificados dependientes de invariantes
C
       if(abs(PROPS(33)-0.0).LT.1.0E-6) then
	   Se=Sv+ Si+ Sa1+ Sa2
	   Ctot=Cv+ Ci+ Ca1+ Ca2
	 elseif(abs(PROPS(33)-1.0).LT.1.0E-6) then
	   call kelvinVoigt(PROPS,NPROPS,DTIME,Hn,Sn,jac,poct,Se,Si,Sa1,
	1   Sa2,Sv,inva,ntens,FG,Ctot,Cv,Ci,Ca1,Ca2)
	 elseif(abs(PROPS(33)-2.0).LT.1.0E-6) then
	   call Maxwell(PROPS,NPROPS,DTIME,Hn,Sn,jac,poct,Se,Si,Sa1,
	1   Sa2,Sv,ntens,FG,Ctot,Cv,Ci,Ca1,Ca2)
	elseif(abs(PROPS(33)-3.0).LT.1.0E-6) then
	   call kelvinVoigt(PROPS,NPROPS,DTIME,Hn,Sn,jac,poct,Se,Si,Sa1,
	1   Sa2,Sv,inva,ntens,FG,Ctot,Cv,Ci,Ca1,Ca2)
	 endif

	 if(abs(PROPS(1)-4.0).LT.1.0E-6) then
	   Ctot=Ctot-derdamM*matmul(Si,transpose(Si))
	 else 
	   Ctot=Ctot-derdamM*matmul(Si,transpose(Si))-
     1	         derdamF1*matmul(Sa1,transpose(Sa1))-
     2		     derdamF2*matmul(Sa2,transpose(Sa2))
       endif
C
C Modificamos el tensor elastico porque ABAQUS trabaja en formulaci�n corrotacional
C
       Hs(1,1)=2.0*Se(1,1); Hs(1,4)=Se(4,1); Hs(1,5)=Se(5,1)
       Hs(2,2)=2.0*Se(2,1); Hs(2,4)=Se(4,1); Hs(2,6)=Se(6,1)
       Hs(3,3)=2.0*Se(3,1); Hs(3,5)=Se(5,1); Hs(3,6)=Se(6,1)
       Hs(4,4)=0.5*(Se(1,1)+Se(2,1)); Hs(4,5)=0.5*Se(6,1)
       Hs(4,6)=0.5*Se(5,1); Hs(5,5)=0.5*(Se(1,1)+Se(3,1))
       Hs(5,6)=0.5*Se(4,1); Hs(6,6)=0.5*(Se(2,1)+Se(3,1))
       do i=1,ntens
         do j=1,i-1
           Hs(i,j)=Hs(j,i)
         enddo
       enddo

       STRESS=Se(:,1)
       DDSDDE=Ctot+Hs
C
C   Almacenamos en las variables de estado el da�o en la matriz y fibras
C   y la funcion energia de deformacion maxima en matriz y fibras en la primera posicion del vector
C
       if(abs(PROPS(1)-4.0).LT.1.0E-6) then
         STATEV(1:6)=damM
         STATEV(14:25)=damF
	 else
         STATEV(1:6)=damM
         STATEV(14:25)=damF
	 endif
	 if (abs(PROPS(1)-8.0).LT.1e-6) then
          STATEV(62:65)=aneurg(1:4)
       endif
C	
C   Almacenamos en las variables de estado Hn y Sn
C
	 STATEV(26:31)=Hn(:,1); STATEV(32:37)=Sn(:,1);
	 STATEV(38:43)=Hn(:,2); STATEV(44:49)=Sn(:,2);
	 STATEV(50:55)=Hn(:,3); STATEV(56:61)=Sn(:,3);

       RETURN
       END SUBROUTINE UMAT
C
C ------------------------------------------------------------------------------
C Subrutina para imporner los desplazamientos para cerrar la
C vena y arteria y luego realizar el grafting
C
C        1         2         3         4         5         6         7
C234567890123456789012345678901234567890123456789012345678901234567890
C ------------------------------------------------------------------------------
      SUBROUTINE  DISP(U,KSTEP,KINC,TIME,NODE,NOEL,JDOF,COORDS)
C
      INCLUDE 'aba_param.inc'
C
       DIMENSION U(3),TIME(2),COORDS(3)
C
       RETURN
      END
C
C ------------------------------------------------------------------------------
C subroutine derW
C ------------------------------------------------------------------------------
C Function to calculate the derivatives of the deviatoric component of the srain 
C energy density function. The following material models are considered:
C
C de(1)  Model
C   0    Isotropic
C   1    Delfino Isotropic
C   2    Holzapfel
C   3    Weiss
C   4    Lin & Yin
C   5    Stochastic damage model
C   6    exponencial
C   7    vangelis
C   8    Vena cava
C   9    Gasser
C   10   AneuGrowthHolz
C
C Input
C
C de   : Vector of amterial properties
C inva : Vector of strain invariants
C dam  : Vector with damage history (For material 5)
C
C Output
C
C w    : Strain energy
C dw1  : First derivative of strain energy with respect invariants
C dw2  : Second derivative of strain energy with respect invariants
C
C Vectors and tensors are organized as follows
C
C dw1=[W1, W2, W4, W6];
C dw2=[W11, W22, W44, W66, W12, W14, W16, W24, W26, W46];
C inva=[I1, I2, I4, I6];
C ------------------------------------------------------------------------------
      subroutine derW(nprop,de,inva,aneurg,w,dw1,dw2)

      integer, intent(in)   :: nprop
      real*8, intent(in)    :: de(nprop),inva(4)
      real*8, intent(in out):: aneurg(4)
      real*8, intent(out)   :: w(4),dw1(4),dw2(10)

      dw1=0.0; dw2=0.0

      if (abs(de(1)) <= 1.0E-6) then  
         call W_Iso(nprop,de,inva,w,dw1,dw2)
      elseif (abs(de(1)-1.0) <= 1.0E-6) then  
         call W_Delfino(nprop,de,inva,w,dw1,dw2)
	elseif (abs(de(1)-2.0) <= 1.0E-6) then  
         call W_Holz(nprop,de,inva,w,dw1,dw2)
      elseif (abs(de(1)-3.0) <= 1.0E-6) then 
         call W_Weiss(nprop,de,inva,w,dw1,dw2)
      elseif (abs(de(1)-4.0)<=1.0E-6) then
         call W_LinYin(nprop,de,inva,w,dw1,dw2)
      elseif (abs(de(1)-5.0)<=1.0E-6) then 
         call W_Stochastic(nprop,de,inva,aneurg,w,dw1,dw2)
      elseif (abs(de(1)-6.0) <= 1.0E-6) then  
         call W_exponential(nprop,de,inva,w,dw1,dw2)
      elseif (abs(de(1)-7.0) <= 1.0E-6) then  
         call W_vangelis(nprop,de,inva,w,dw1,dw2)
	elseif (abs(de(1)-8.0) <= 1.0E-6) then  
	   call W_New_VC(nprop,de,inva,w,dw1,dw2)
	elseif (abs(de(1)-9.0) <= 1.0E-6) then  
	   call W_Gasser(nprop,de,inva,w,dw1,dw2)
	elseif (abs(de(1)-10.0) <= 1.0E-6) then  
	   call W_AneuGrowthHolz(nprop,de,inva,aneurg,w,dw1,dw2)
      endif
       end subroutine derW
C ------------------------------------------------------------------------------
      subroutine W_Iso(nprop,de,inva,w,dw1,dw2)
C ------------------------------------------------------------------------------
C
C Evaluates the Isotropic Strain energy density function
C and its derivative
C
C---------------------------------------------------
C
C de      ... array of material properties
C inv     ... array with invariants
C dam     ... auxiliary vector -notused-
C w       ... Strain energy
C dw1     ... First derivative of the Strain energy
C dw2     ... Second derivative of the Strain energy
C---------------------------------------------------
      INCLUDE 'aba_param.inc'
      integer, intent(in)   :: nprop
      real*8, intent(in)    :: de(nprop),inva(4)
      real*8, intent(out)   :: w(4),dw1(4),dw2(10)
      real*8                :: C10,C01,C20,C11,C02
      real*8                :: I1,I2

      C10=de(3); C01=de(4); C20=de(5); C11=de(6); C02=de(7);
      I1=inva(1); I2=inva(2)
C
C Strain energy density function
C
      w(2)= C10*(I1-3.0)+C01*(I2-3.0)+C20*(I1-3.0)**2  
      w(2)= w(2)+C11*(I1-3.0)*(I2-3.0)+C02*(I2-3.0)**2
	w(1)= w(2) 
C
C First derivative
C     
      dw1(1)=C10+C11*(I2-3.0)+2.0*C20*(I1-3.0)
      dw1(2)=C01+C11*(I1-3.0)+2.0*C02*(I2-3.0)
C
C Second derivative
C     
      dw2(1)=2*C20
      dw2(2)=2*C02
      dw2(5)=C11
      end subroutine
C --------------------------------------------------------------------
      subroutine W_Delfino(nprop,de,inva,w,dw1,dw2)
C --------------------------------------------------------------------
C
C Evaluates the Demiray isotropic strain energy function and its derivative
C
C --------------------------------------------------------------------
      INCLUDE 'aba_param.inc'
      integer, intent(in)   :: nprop
      real*8, intent(in)    :: de(nprop),inva(4)
      real*8, intent(out)   :: w,dw1(4),dw2(10)
      real*8                :: a,b,expo
      real*8                :: I1

      a=de(3); b=de(4)
      I1=inva(1)
C
C Strain energy density function
C
      expo=exp(0.5*b*(I1-3.0))
      w= (a/b)*(expo-1.0) 
C
C First derivative
C     
      dw1(1)=0.5*a*expo
C
C Second derivative
C     
      dw2(1)=0.5*b*dw1(1)
      end subroutine
C ------------------------------------------------------------------------------
      subroutine W_Holz(nprop,de,inva,w,dw1,dw2)
C ------------------------------------------------------------------------------
C
C Evaluates the Holzapfel's Strain energy density function and its derivative
C
C----------------------------------------------------------------
C
C de      ... array of material properties
C inv     ... array with invariants
C dam     ... auxiliary vector -not used-
C w       ... Strain energy
C dw1     ... First derivative of the Strain energy
C dw2     ... Second derivative of the Strain energy
C----------------------------------------------------------------
      INCLUDE 'aba_param.inc'
      integer, intent(in)   :: nprop
      real*8, intent(in)    :: de(nprop),inva(4)
      real*8, intent(out)   :: w(4),dw1(4),dw2(10)
      real*8                :: C10,C01,C20,C11,C02
      real*8                :: I1,I2,I4,I6,stretcha,stretchb
      real*8                :: k1,k2,k3,k4,rho
      real*8                :: term1,term4,term6,exp1,exp2
      real*8                :: expon11,expon14,expon21,expon26
C
C Material properties
C
      C10=de(3); C01=de(4); C20=de(5); C11=de(6); C02=de(7);
      I1=inva(1); I2=inva(2); I4=inva(3); I6=inva(4);
      stretcha = sqrt(I4)
      stretchb = sqrt(I6) 
      k1 = de(8); k2 = de(9);rho = de(12)
	lam0 = sqrt(de(13)); I40=de(13)
      if (abs(k1*k2) .lt. 1E-8) then
          k1=0.0; k2=1.0
      endif
      k3 = de(10); k4 = de(11) 
      if (abs(k3*k4) .lt. 1E-8) then
          k3=0.0; k4=1.0
      endif
C
C  Main terms in the strain energy function
C
      expon11=k2*(1.0-rho)*(I1-3.0)*(I1-3.0)
      expon21=k4*(1.0-rho)*(I1-3.0)*(I1-3.0)
      term1=(I1-3.)*(1.-rho)
      expon14=0.0; term4=0.0
      if (stretcha .ge. lam0) then
        expon14=k2*rho*(I4-I40)*(I4-I40)
        term4=rho*(I4-I40)
      endif
      expon26=0.0; term6=0.0
      if (stretchb .ge. lam0) then
        expon26=k4*rho*(I6-I40)*(I6-I40)
        term6=rho*(I6-I40)
      endif
      exp1=exp(expon11+expon14)
      exp2=exp(expon21+expon26)
C
C Strain energy density function 
C Almacenamos la enrgia total en w(1), la de la matriz en w(2) y la de las fibras en w(3) y w(4)
C
         w(2)= C10*(I1-3.0)+C01*(I2-3.0)+C20*(I1-3.0)**2
         w(2)= w(2)+C11*(I1-3.0)*(I2-3.0)+C02*(I2-3.0)**2
         w(3)= (k1/k2)*(exp1-1.0)
	   w(4)= (k3/k4)*(exp2-1.0)
	   w(1) =w(2)+w(3)+w(4)
C
C First derivative
C     
      dw1(1)=C10+C11*(I2-3.0)+2.0*C20*(I1-3.0)
      dw1(1)=dw1(1)+term1*(k1*exp1+k3*exp2)
      dw1(2)=C01+C11*(I1-3.0)+2.0*C02*(I2-3.0)
      dw1(3)=k1*term4*exp1
      dw1(4)=k3*term6*exp2
C
C Second derivative
C     
      dw2(1)=2*C20+k1*((1.-rho)+2.*k2*term1*term1)*exp1
      dw2(1)=dw2(1)+k3*((1.-rho)+2.*k4*term1*term1)*exp2
      dw2(2)=2*C02
      if (stretcha .gt. lam0) then
        dw2(3)=k1*(rho+2.*k2*term4*term4)*exp1
        dw2(6)=2.*k1*k2*term1*term4*exp1
      endif
      if (stretchb .gt. lam0) then
        dw2(4)=k3*(rho+2.*k4*term6*term6)*exp2
        dw2(7)=2.*k3*k4*term1*term6*exp2
      endif
      dw2(5)=C11
      end subroutine
C ------------------------------------------------------------------------------
      subroutine W_Weiss(nprop,de,inva,w,dw1,dw2)
C ------------------------------------------------------------------------------
C
C Evaluates the Weiss Strain energy density function and its derivative
C
C----------------------------------------------------------------------
C
C de      ... array of material properties
C inv     ... array with invariants
C dam     ... auxiliary vector -notused-
C w       ... Strain energy
C dw1     ... First derivative of the Strain energy
C dw2     ... Second derivative of the Strain energy
C----------------------------------------------------------------------
      INCLUDE 'aba_param.inc'
      integer, intent(in)   :: nprop
      real*8, intent(in)    :: de(nprop),inva(4)
C      real*8, intent(in out):: dam(4)
      real*8, intent(out)   :: w(4),dw1(4),dw2(10)
      real*8                :: C10,C01,C20,C11,C02
      real*8                :: stretch,stretchref
      real*8                :: k1,k2,k3,k4
      real*8                :: I1,I2,I4

      C10=de(3); C01=de(4); C20=de(5); C11=de(6); C02=de(7);
      I1=inva(1); I2=inva(2); I4=inva(3)
C
C  Isotropic contribution
C    
         w(2)= C10*(I1-3.0)+C01*(I2-3.0)+C20*(I1-3.0)**2
         w(2)= w(2)+C11*(I1-3.0)*(I2-3.0)+C02*(I2-3.0)**2
C    
C First derivative
C         
         dw1(1)=C10+C11*(I2-3.0)+2.0*C20*(I1-3.0)
         dw1(2)=C01+C11*(I1-3.0)+2.0*C02*(I2-3.0)
C    
C Second derivative
C        
         dw2(1)=2*C20
         dw2(2)=2*C02
         dw2(5)=C11
    
         stretch = sqrt(I4)
         stretchref = de(8)
         k1=de(9); k2=de(10)
         k3=de(11); k4=de(12)
       
         if ((stretch - 1.0) < -1.0e-8) then
           dw1(2) = 0.0
           dw2(8) = 0.0          
         elseif (stretch  <= stretchref) then
           dw1(3) = (k1 * (exp(k2*(stretch-1.0)) - 1.0)) / (2.0 * I4)
           dw2(3)=(k1*k2*stretch*(exp(k2*(stretch-1.0))) - 
     1         2.0*k1*(exp(k2*(stretch-1.0)) - 1.0)) / (4.0*I4*I4)
         elseif (stretch  > stretchref) then
           w(3)=k3*stretch + k4*log(stretch) 
           dw1(3) = (k3*stretch + k4) / (2.0*I4)
           dw2(3) =(-1.0) * (k3*stretch + 2.0*k4) / (4.0*I4*I4)
         endif
	   	 w(1)= w(2)+w(3)
      end subroutine
C ------------------------------------------------------------------------------
      subroutine W_LinYin(nprop,de,inva,w,dw1,dw2)
C ------------------------------------------------------------------------------
C
C Evaluates the Lin and Yin Strain energy density function and its derivative
C
C--------------------------------------------------------------------
C
C de      ... array of material properties
C inv     ... array with invariants
C dam     ... auxiliary vector -notused-
C w       ... Strain energy
C dw1     ... First derivative of the Strain energy
C dw2     ... Second derivative of the Strain energy
C--------------------------------------------------------------------
      INCLUDE 'aba_param.inc'
      integer, intent(in)   :: nprop
      real*8, intent(in)    :: de(nprop),inva(4)
      real*8, intent(out)   :: w(4),dw1(4),dw2(10)
      real*8                :: I1,I2,I4
      real*8                :: C1P,C2P,C3P,C4P
      real*8                :: beta,C1A,C2A,C3A,C4A,C5A
      real*8                :: Q,dQI1,dQI4,ter1
C
      C10=de(3); C01=de(4); C20=de(5); C11=de(6); C02=de(7);
      I1=inva(1); I2=inva(2); I4=inva(3)
C
      C1P=de(3); C2P=de(4); C3P=de(5); C4P=de(6)
      beta=de(7); 
      C1A=de(8); C2A=de(9); C3A=de(10); C4A=de(11); C5A=de(12)
    
      Q=C2P*(I1-3.0)*(I1-3.0) + C3P*(I1-3.0)*(I4-1.0) + 
     1    C4P*(I4-1.0)*(I4-1.0)
      dQI1=2.0*C2P*(I1-3.0) + C3P*(I4-1.0)
      dQI4=C3P*(I1-3.0) + 2.0*C4P*(I4-1.0)
      ter1=C1P*EXP(Q)
	w(2)=ter1-C1P
	w(3)=beta*(C1A*(I4-1.0)*(I1-3)+C2A*(I1-3.0)**2+
     1  C3A*(I4-1.0)**2+C4A*(I1-3.0)+C5A*(I4-1.0))
C
C       Strain energy
C
      w(1)= ter1-C1P+beta*(C1A*(I4-1.0)*(I1-3)+C2A*(I1-3.0)**2+
     1  C3A*(I4-1.0)**2+C4A*(I1-3.0)+C5A*(I4-1.0))
C    
C       First derivative
C        
      dw1(1)=ter1*dQI1 + beta*(C1A*(I4-1.0)+2.0*C2A*(I1-3.0)+C4A)
      dw1(2)=0.0
      dw1(3)=ter1*dQI4 + beta*(C1A*(I1-3.0)+2.0*C3A*(I4-1.0)+C5A)
      dw1(4)=0.0
C        
C       Second derivative
C        
      dw2(1)=ter1*(dQI1*dQI1+2.0*C2P) + beta*2.0*C2A
      dw2(3)=ter1*(dQI4*dQI4+2.0*C4P) + beta*2.0*C3A
      dw2(6)=ter1*(dQI1*dQI4+C3P) + beta*C1A
      end subroutine
C ------------------------------------------------------------------------------
      subroutine W_Stochastic(nprop,de,inva,dam,w,dw1,dw2)
C ------------------------------------------------------------------------------
C
C Evaluates the Stochastic Strain energy density function and its derivative
C
C---------------------------------------------------------------
C
C de      ... array of material properties
C inv     ... array with invariants
C dam     ... auxiliary vector -damage variables are stored-
C w       ... Strain energy
C dw1     ... First derivative of the Strain energy
C dw2     ... Second derivative of the Strain energy
C---------------------------------------------------------------
      INCLUDE 'aba_param.inc'
      integer, intent(in)   :: nprop
      real*8, intent(in)    :: de(nprop),inva(4)
      real*8, intent(in out):: dam(4)
      real*8, intent(out)   :: w(4),dw1(4),dw2(10)
      real*8                :: C10,C01,C20,C11,C02
      real*8                :: I1,I2,I4,I6
      real*8                :: stretcha, stretchb
      real*8                :: C1,kappa,eta,gama,r0,bet,delta
      real*8                :: r,emaxA,emaxB,D11,D12,D13,D1
      real*8                :: x1,x2,a1,b1,a2,b2
      real*8                :: ter11,ter2,ter3,Integ

      C10=de(3); C01=de(4); C20=de(5); C11=de(6); C02=de(7);
      I1=inva(1); I2=inva(2); I4=inva(3); I6=inva(4);
C
C Isotropic contribution
C         
        w(2)= C10*(I1-3.0)+C01*(I2-3.0)+C20*(I1-3.0)**2 
        w(2)= w(2)+C11*(I1-3.0)*(I2-3.0)+C02*(I2-3.0)**2
	  w(1)= w(2)
C
C     First derivative
C        
        dw1(1)=C10+C11*(I2-3.0)+2.0*C20*(I1-3.0)
        dw1(2)=C01+C11*(I1-3.0)+2.0*C02*(I2-3.0)
C        
C     Second derivative
C        
        dw2(1)=2*C20
        dw2(2)=2*C02
        dw2(5)=C11
C    
C Anisotropic contribution
C    
        stretcha = sqrt(I4)
        stretchb = sqrt(I6)
C
C   Model parameters
C
        D1 = de(11)
        gama = de(12)
        eta =de(13)
        kappa = de(14)
        r0=de(15)
        bet=de(16)
        delta=de(17)
C
C   Max. strain history
C
        emaxA=dam(1)
        emaxB=dam(3)
C
        eta = (kappa - mu1)*(kappa*mu1 - mu1*mu1 - 
     1	  sig1*sig1)/(kappa*sig1*sig1)
        gama = mu1*(kappa*mu1 - mu1*mu1 - sig1*sig1)/(kappa*sig1*sig1)
        D11=gammaln(gama)
        D12=gammaln(eta)
        D13=gammaln(gama+eta)
        D1=0.25*C1*exp(D13-D11-D12)
C
C Calculating derivatives
C
C First family of fibers
C
        es=strecha-1.0
        x2=kappa
        if(es > 0.0) then
          if(abs(es-kappa/bet)<1E-3) then
            dw1(3)=0.0
            dw2(3)=0.0
          else
            if(es>=emaxA) then
              emaxA=es
              x1=es*bet
              a1=x1/x2
              b1=gama-1
              a2=1-a1
              b2=eta-1
              ter11=(bet-4.0-bet*bet*bet/((bet-1)*(bet-1)))*
     1			(a1**b1)*(a2**b2)
            else
              x1=emaxA*bet
              ter11=0.0
            end if

C First derivative

            Integ=integrate(x1,x2,es,kappa,eta,gama,D1,1)
C            ter1=betai(gama, eta, x1/kappa)
C            ter2=betai(gama-1.0, eta, x1/kappa)
            ter3=(gama+eta-1.0)/(gama-1.0)

C
C Updating amount of damage
C
            dam(2)=ter1
            dw1(3)=0.5*(C1*(0.25*ter1-0.25+(es/kappa)*ter3*(1.0-ter2))+
     1		  Integ)/strecha
            if (dw1(3)<0.0) then
               dw1(3)=0.0
            end if
C
C Second derivative
C
            Integ=integrate(x1,x2,es,kappa,eta,gama,D1,2)
            dw2(3)=(0.25/I4)*(D1*ter11/kappa+C1*ter3*(1.0-ter2)/kappa+
	1	  Integ - 2.0*dw1(3))
          end if
        end if
C
C Second family of fibers
C
        es=strechb-1.0
        if(es > 0.0) then
          if(abs(es-kappa/bet)<1E-3) then
            dw1(3)=0.0
            dw2(9)=0.0
          else
             if(es>=emaxB) then
              emaxB=es
              x1=es*bet
              a1=x1/x2
              b1=gama-1
              a2=1-a1
              b2=eta-1
              ter11=(bet-4.0-bet*bet*bet/((bet-1)*(bet-1)))*
     1			(a1**b1)*(a2**b2)
             else
              x1=emaxB*bet
              ter11=0.0
             end if
C
C First derivative
C
             Integ=integrate(x1,x2,es,kappa,eta,gama,D1,1)
C             ter1=betai(gama, eta, x1/kappa)
C             ter2=betai(gama-1.0, eta, x1/kappa)
             ter3=(gama+eta-1)/(gama-1)
C
C Updating amount of damage
C
             dam(4)=ter1
             dw1(4)=0.5*(C1*(0.25*ter1-0.25+(es/kappa)*ter3*(1.0-ter2))
     1		   +Integ)/strechb
             if (dw1(4)<0.0) then
               dw1(4)=0.0
             end if
C
C Second derivative
C
             Integ=integrate(x1,x2,es,kappa,eta,gama,D1,2)
             dw2(4)=(0.25/I4)*(D1*ter11/kappa+C1*ter3*(1.0-ter2)/kappa+
     1		   Integ - 2.0*dw1(4))
         end if
        end if
        dam(1)=emaxA
        dam(3)=emaxB
      end subroutine
C ------------------------------------------------------------------------------
      subroutine W_exponential(nprop,de,inva,w,dw1,dw2)
C ------------------------------------------------------------------------------
C
C Evaluates the exponential energy density function and its derivative
C
C---------------------------------------------------------------
C
C de      ... array of material properties
C inv     ... array with invariants
C dam     ... auxiliary vector -notused-
C w       ... Strain energy
C dw1     ... First derivative of the Strain energy
C dw2     ... Second derivative of the Strain energy
C---------------------------------------------------------------
      INCLUDE 'aba_param.inc'
      integer, intent(in)   :: nprop
      real*8, intent(in)    :: de(nprop),inva(4)
      real*8, intent(out)   :: w(4),dw1(4),dw2(10)
      real*8                :: C10,C01,C20,C11,C02
      real*8                :: stretch,stretch0
      real*8                :: a,b,k1,k2,k3,k4
      real*8                :: I1,I2,I4,I6,I40,I60

      C10=de(3); C01=de(4); C20=de(5); C11=de(6); C02=de(7);
      I1=inva(1); I2=inva(2); I4=inva(3); I6=inva(4)
	a=de(8); b=de(9);
	k1=de(10); k2=de(11); k3=de(12); k4=de(13); 
	I40=de(14); I60=de(15);
	if (abs(k1*k2) .lt. 1E-8) then
          k1=0.0; k2=1.0
      endif
      if (abs(k3*k4) .lt. 1E-8) then
          k3=0.0; k4=1.0
      endif
C
C  Isotropic contribution
C    
         expo=exp(0.5*b*(I1-3.0))
	   w(2)= C10*(I1-3.0)+C01*(I2-3.0)+C20*(I1-3.0)**2
         w(2)= w(2)+C11*(I1-3.0)*(I2-3.0)+C02*(I2-3.0)**2
	   if (b > 1.0e-8) then
	   w(2)= w(2)+(a/b)*(expo-1.0)
	   endif
C    
C First derivative
C         
         dw1(1)=C10+C11*(I2-3.0)+2.0*C20*(I1-3.0)+0.5*a*expo
         dw1(2)=C01+C11*(I1-3.0)+2.0*C02*(I2-3.0)
C    
C Second derivative
C        
         dw2(1)=2*C20
         dw2(2)=2*C02
         dw2(5)=C11+0.5*b*0.5*a*expo
    
         stretcha = sqrt(I4); stretcha0 = sqrt(I40)
	   if ((stretcha - stretcha0) < -1.0e-8) then
           dw1(3) = 0.0
           dw2(3) = 0.0
		 w(3)=0.0
         else
           dw1(3) = k1 * (exp(k2*(I4-I40))-1)
           dw2(3)=k1*k2*(exp(k2*(I4-I40)))
		 w(3)= k1/k2 * (exp(k2*(I4-I40))-k2*(I4-I40)-1)
         endif

         stretchb = sqrt(I6); stretchb0 = sqrt(I60)
	   if ((stretchb - stretchb0) < -1.0e-8) then
           dw1(4) = 0.0
           dw2(4) = 0.0
		 w(4)=0.0
         else
! Modified with k3 and k4
           dw1(4) = k3 * (exp(k4*(I6-I60))-1)
           dw2(4)=k3*k4*(exp(k4*(I6-I60)))
		 w(4)= k3/k4 * (exp(k4*(I6-I60))-k4*(I6-I60)-1)
         endif
	   w(1)=w(2)+w(3)+w(4)
      end subroutine
C ------------------------------------------------------------------------------
      subroutine W_vangelis(nprop,de,inva,w,dw1,dw2)
C ------------------------------------------------------------------------------
C
C Evaluates the Vangelis Strain energy density function
C and its derivative. This is a modified Weiss strain energy function
C
C--------------------------------------------------------------
C
C de      ... array of material properties
C inv     ... array with invariants
C dam     ... auxiliary vector -notused-
C w       ... Strain energy
C dw1     ... First derivative of the Strain energy
C dw2     ... Second derivative of the Strain energy
C--------------------------------------------------------------
      INCLUDE 'aba_param.inc'
      integer, intent(in)   :: nprop
      real*8, intent(in)    :: de(nprop),inva(4)
      real*8, intent(out)   :: w(4),dw1(4),dw2(10)
      real*8                :: C10,C01,C20,C11,C02
      real*8                :: stretch,stretchref,stretch0
      real*8                :: k1,k2,k3,k4,k5
      real*8                :: I1,I2,I4,I40,I4ref

		C10=de(3); C01=de(4); C20=de(5); C11=de(6); C02=de(7);
		a=de(8); b=de(9);k1=de(10); k2=de(11); 
	    I40=de(12);I4ref=de(13)
		I1=inva(1); I2=inva(2); I4=inva(3)
C
C  Isotropic contribution
C    
         expo=exp(0.5*b*(I1-3.0))
	   w(2)= C10*(I1-3.0)+C01*(I2-3.0)+C20*(I1-3.0)**2
         w(2)= w(2)+C11*(I1-3.0)*(I2-3.0)+C02*(I2-3.0)**2
	   if (b > 1.0e-8) then
	   w(2)= w(2)+(a/b)*(expo-1.0)
	   endif
C    
C First derivative
C         
         dw1(1)=C10+C11*(I2-3.0)+2.0*C20*(I1-3.0)+0.5*a*expo
         dw1(2)=C01+C11*(I1-3.0)+2.0*C02*(I2-3.0)
C    
C Second derivative
C        
         dw2(1)=2*C20
         dw2(2)=2*C02
         dw2(5)=C11+0.5*b*0.5*a*expo
    
         stretch = sqrt(I4)
	   stretch0 = sqrt(I40); stretchref = sqrt(I4ref)

	   k3=4.0*k1*stretchref*(exp(k2*(I4ref-I40))+
	1      k2*I4ref*exp(k2*(I4ref-I40))-1)
	   k4=-2.0*k1*I4ref*(exp(k2*(I4ref-I40))+
	1      2.0*k2*I4ref*exp(k2*(I4ref-I40))-1)

	   k5=(1.0/k2)*k1*(k2*I40+3*k2*I4ref-
	1      1+exp(k2*(I4ref-I40))*(1-4.0*k2*I4ref-4.0*k2**2*I4ref**2)+
     2      k2*I4ref*log(I4ref)*(exp(k2*(I4ref-I40))*(1+2*k2*I4ref)
     3      -1))
	   if ((stretch - stretch0) < -1.0e-8) then
           dw1(2) = 0.0
           dw2(8) = 0.0
		 w(3)=0.0
         elseif (stretch  <= stretchref) then
           dw1(3) = k1 * (exp(k2*(I4-I40))-1)
           dw2(3)=k1*k2*(exp(k2*(I4-I40)))
		 w(3)= k1/k2 * (exp(k2*(I4-I40))-k2*(I4-I40)-1)
C	   elseif (stretch  > stretchref) then
	   else
	     w(3)=k3*stretch +0.5* k4*log(I4) + k5
		 dw1(3) = (k3*stretch + k4) / (2.0*I4)
           dw2(3) =(-1.0)*(k3*stretch + 2.0*k4)/(4.0*I4*I4)
         endif
	   w(1)=w(2)+w(3)
      end subroutine
C ------------------------------------------------------------------------------
      subroutine W_New_VC(nprop,de,inva,w,dw1,dw2)
C ------------------------------------------------------------------------------
C
C Evaluates the New Vena Cava Strain energy density function
C and its derivative
C
C---------------------------------------------------
C Input
C
C de   : Vector of material properties
C inva : Vector of strain invariants
C dam  : Vector with damage history (For material 4)
C
C Output
C
C w    : Strain energy
C dw1  : First derivative of strain energy with respect invariants
C dw2  : Second derivative of strain energy with respect invariants
C
C Vectors and tensors are organized as follows
C
C dw1=[W1, W2, W4, W6];
C dw2=[W11, W22, W44, W66, W12, W14, W16, W24, W26, W46];
C inva=[I1, I2, I4, I6];
C---------------------------------------------------
      INCLUDE 'aba_param.inc'
      integer, intent(in)   :: nprop
      real*8, intent(in)    :: de(nprop),inva(4)
C      real*8, intent(inout) :: sdv(nstatv)
      real*8, intent(out)   :: w,dw1(4),dw2(10)
      real*8                :: C10,C2,C3
      real*8                :: I1,I4,I6
      real*8                :: stretchI4, stretchI6, invstrI4, invstrI6
      real*8                :: phi_A, phi_I

      C10=de(3); C2=de(8); C3=de(9); 
      I1=inva(1); I4=inva(3); I6=inva(4);
      phi_I=de(10) 
      phi_A=de(11)

C Strain energy density function

       w = 0.0
       dw1=0.0
       dw2=0.0
       stretchI4 = sqrt(I4)
       stretchI6 = sqrt(I6)

       w=phi_I*C10*(I1-3.0)
       dw1(1)=phi_I*C10
       dw2(1)=0.0

       if (stretchI4.gt.1.0) then
  	  w=w+phi_A*C2*(stretchI4-1)**2
          invstrI4=1.0/stretchI4 
	  dw1(3)=phi_A*(C2+C2*invstrI4)
	  dw2(3)=0.5*phi_A*C2*invstrI4**3
       end if

       if (stretchI6.gt.1.0) then
	  w=w+phi_A*C3*(stretchI6-1)**2      
          invstrI6=1.0/stretchI6 
	  dw1(4)=phi_A*(C3+C3*invstrI6)
	  dw2(4)=0.5*phi_A*C3*invstrI6**3
       end if
      end subroutine W_New_VC
C ------------------------------------------------------------------------------
      subroutine W_Gasser(nprop,de,inva,w,dw1,dw2)
C ------------------------------------------------------------------------------
C
C Evaluates the Gasser's Strain energy density function and its derivative
C
C----------------------------------------------------------------
C
C de      ... array of material properties
C inv     ... array with invariants
C dam     ... auxiliary vector -not used-
C w       ... Strain energy
C dw1     ... First derivative of the Strain energy
C dw2     ... Second derivative of the Strain energy
C
C Vectors and tensors are organized as follows
C
C dw1=[W1, W2, W4, W6];
C dw2=[W11, W22, W44, W66, W12, W14, W16, W24, W26, W46];
C inva=[I1, I2, I4, I6];
C----------------------------------------------------------------
      INCLUDE 'aba_param.inc'
      integer, intent(in)   :: nprop
      real*8, intent(in)    :: de(nprop),inva(4)
      real*8, intent(out)   :: w(4),dw1(4),dw2(10)
      real*8                :: C10,C01,C20,C11,C02
      real*8                :: I1,I2,I4,I6,stretcha,stretchb
      real*8                :: I40,I60,lam40,lam60
      real*8                :: k1,k2,kappa,expon4,expon6
      real*8                :: term1,term4,term6,exp1,exp2
C
C Material properties
C
      C10=de(3); C01=de(4); C20=de(5); C11=de(6); C02=de(7);
      I1=inva(1); I2=inva(2); I4=inva(3); I6=inva(4);
      stretcha = sqrt(I4)
      stretchb = sqrt(I6) 
      k1 = de(8); k2 = de(9);kappa = de(12)
	I40=de(13);lam40=sqrt(I40)
	I60=de(14);lam60=sqrt(I60)
      if (abs(k1*k2) .lt. 1E-8) then
          k1=0.0; k2=1.0
      endif
C
C  Main terms in the strain energy function
C
      term1=0.0; term4=0.0; term6=0.0
      term1=kappa*(I1-3.0)
      if (stretcha .ge. lam40) then
        term4=(1.-3.0*kappa)*(I4-I40)
      endif
      if (stretchb .ge. lam60) then
        term6=(1.-3.0*kappa)*(I6-I60)
      endif
      expon4=0.0;expon6=0.0;exp1=0.0;exp2=0.0
      expon4=(term1+term4)**2
      expon6=(term1+term6)**2
      exp1=exp(k2*expon4)
      exp2=exp(k2*expon6)
C
C Strain energy density function 
C Almacenamos la enrgia total en w(1), la de la matriz en w(2) y la de las fibras en w(3) y w(4)
C
       w(2)= C10*(I1-3.0)+C01*(I2-3.0)+C20*(I1-3.0)**2
       w(2)= w(2)+C11*(I1-3.0)*(I2-3.0)+C02*(I2-3.0)**2
       w(3)= (k1/(2*k2))*(exp1-1.0)
	 w(4)= (k1/(2*k2))*(exp2-1.0)
	 w(1) =w(2)+w(3)+w(4)
C
C First derivative
C     
      dw1(1)=C10+C11*(I2-3.0)+2.0*C20*(I1-3.0)
      dw1(1)=dw1(1)+k1*kappa*exp1*(term1+term4)
      dw1(1)=dw1(1)+k1*kappa*exp2*(term1+term6)
      dw1(2)=C01+C11*(I1-3.0)+2.0*C02*(I2-3.0)
      dw1(3)=k1*(1.-3.0*kappa)*exp1*(term1+term4)
      dw1(4)=k1*(1.-3.0*kappa)*exp2*(term1+term6)
C
C Second derivative
C     
      dw2(1)=2*C20+k1*kappa*kappa*exp1*(1+2.*k2*(term1+term4)**2)
      dw2(1)=dw2(1)+k1*kappa*kappa*exp2*(1+2.*k2*(term1+term6)**2)
      dw2(2)=2*C02
      if (stretcha .gt. lam40) then
        dw2(3)=k1*((1.-3.0*kappa)**2)*exp1*(1+2.*k2*(term1+term4)**2)
        dw2(6)=k1*(1.-3.0*kappa)*kappa*exp1*(1+2.*k2*(term1+term4)**2)
      endif
      if (stretchb .gt. lam60) then
        dw2(4)=k1*((1.-3.0*kappa)**2)*exp2*(1+2.*k2*(term1+term6)**2)
        dw2(7)=k1*(1.-3.0*kappa)*kappa*exp2*(1+2.*k2*(term1+term6)**2)
      endif
      dw2(5)=C11
      end subroutine W_Gasser
C ------------------------------------------------------------------------------
      subroutine W_AneuGrowthHolz(nprop,de,inva,dam,w,dw1,dw2)
C ------------------------------------------------------------------------------
C
C Evaluates the Holzapfel Strain energy density function for aneurism growth and 
C its derivative
C
C------------------------------------------------------------------
C
C de      ... array of material properties
C inv     ... array with invariants
C dam     ... auxiliary vector -plastic strain and fiber stress-
C w       ... Strain energy
C dw1     ... First derivative of the Strain energy
C dw2     ... Second derivative of the Strain energy
C------------------------------------------------------------------
      INCLUDE 'aba_param.inc'
      integer, intent(in)   :: nprop
      real*8, intent(in)    :: de(nprop),inva(4)
      real*8, intent(in out):: dam(4)
      real*8, intent(out)   :: w(4),dw1(4),dw2(10)
      real*8                :: C10,C01,C20,C11,C02
      real*8                :: I1,I2,I4,I6,stretcha,stretchb
      real*8                :: k1,k2,k3,k4,rho
      real*8                :: term1,term4,term6,exp1,exp2
      real*8                :: expon11,expon14,expon21,expon26
C
C Material properties
C
      C10=de(3); C01=de(4); C20=de(5); C11=de(6); C02=de(7);
      I1=inva(1); I2=inva(2)
C
C Material properties
C
        stretcha = sqrt(I4)-dam(1)
        if(stretcha .lt. 1.0) then
          stretcha=1.0
        endif
        if(stretchb .lt. 1.0) then
          stretchb = sqrt(I6)-dam(3) 
        endif
        k1 = de(8); k2 = de(9)  
        if ((abs(k1*k2) .lt. 1E-8).or.(stretcha .le. 1.0)) then
            k1=0.0; k2=1.0
        endif
        k3 = de(10); k4 = de(11) 
        if ((abs(k3*k4) .lt. 1E-8).or.(stretchb .le. 1.0)) then
            k3=0.0; k4=1.0
        endif
C  Main terms in the strain energy function
        expon14=0.0; term4=0.0
        if (stretcha .gt. 1.0) then
          expon14=k2*(I4-1.0)*(I4-1.0)
          term4=(I4-1.0)
        endif
        expon26=0.0; term6=0.0
        if (stretchb .gt. 1.0) then
          expon26=k4*(I6-1.0)*(I6-1.0)
          term6=(I6-1.0)
        endif
        exp1=exp(expon14)
        exp2=exp(expon26)

C Strain energy density function and derivatives
         w(2)= C10*(I1-3.0)+C01*(I2-3.0)+C20*(I1-3.0)**2  
         w(2)= w(2)+C11*(I1-3.0)*(I2-3.0)+C02*(I2-3.0)**2 
         w(3)= (k1/k2)*(exp1-1.0)
	   w(4)= (k3/k4)*(exp2-1.0)
	   w(1)= w(2) + w(3)+ w(4)

C First derivative
     
         dw1(1)=C10+C11*(I2-3.0)+2.0*C20*(I1-3.0)
         dw1(2)=C01+C11*(I1-3.0)+2.0*C02*(I2-3.0)
         dw1(3)=k1*term4*exp1
         dw1(4)=k3*term6*exp2

C Second derivative
     
         dw2(1)=2*C20
         dw2(2)=2*C02
         if (stretcha .gt. 1.0) then
           dw2(3)=k1*(1.0+2.*k2*term4*term4)*exp1
         endif
         if (stretchb .gt. 1.0) then
           dw2(4)=k3*(1.0+2.*k4*term6*term6)*exp2
         endif
         dw2(5)=C11

         dam(2)=2*stretcha*dw1(3)
         dam(4)=2*stretchb*dw1(4)

      end subroutine
C ------------------------------------------------------------------------------
      subroutine dam_iso_func(nprop,de,inv,damM,derdamM)
C ------------------------------------------------------------------------------
C
C Evaluates the isotropic damage function and its derivative, only valid 
C to W_Stochastic
C
C-------------------------------------------------------------
C
C de      ... array of material properties
C inv     ... array with invariants
C damM(1) ... damage strain for damage evaluation
C damM(2) ... Damage function
C derDam_M... Derivative of damage function
C-------------------------------------------------------------
        INCLUDE 'aba_param.inc'

        integer, intent(in)   :: nprop
        real*8, intent(in)    :: de(nprop),inv(4)
        real*8, intent(in out):: damM(2)
        real*8, intent(out)   :: derdamM
        real*8                :: C10,C01,C20,C11,C02,alf,bet,E0
        real*8                :: I1,I2,I4,I6,Phi,etrial
        real*8                :: expon,ter1,ter2,ter3
C
C Strain energy density function
C
        C10=de(3); C01=de(4); C20=de(5); C11=de(6); C02=de(7);
        alf=de(8); bet=de(9); E0=de(10);
        I1=inv(1); I2=inv(2); I4=inv(3); I6=inv(4);

        Phi= C10*(I1-3.0)+C01*(I2-3.0)+C20*(I1-3.0)**2
        Phi= Phi+C11*(I1-3.0)*(I2-3.0)+C02*(I2-3.0)**2
        etrial=sqrt(Phi/de(10))
        derdamM=0.0;

        if(( etrial > damM(1)) .AND. bet>0) then
         damM(1)=etrial
         expon=2.0*alf*(2.0*etrial/bet-1.0)
         ter1=exp(expon)
         ter2=etrial*alf*exp(expon)-1.0
         ter3=1.0/(2+ter2)
         damM(2)=0.5*(1.0+ter2*ter3)
         derDamM=2.0*(1-damM(2))*(alf/bet)*(4.0*etrial*alf+bet)
	1	   *ter1*ter3
        end if
      end subroutine dam_iso_func
C ------------------------------------------------------------------------------
      function integrate(a,b,es,kappa,eta,gama,C1,tipo)
C ------------------------------------------------------------------------------
C Function to performe numerical integration
C for the stochastic strain energy function
C with damage
C ------------------------------------------------------------------------------
        INCLUDE 'aba_param.inc'

        real*8,parameter   :: EPSIL= 3.0e-7
        integer,parameter  :: NDIV = 12
        real*8, intent(in) :: a,b,es,kappa,eta,gama,C1
        integer,intent(in) :: tipo
        real*8             :: integrate,x1,x2,dx,ba
        integer            :: i

        ba = b-EPSIL;
        dx=(ba-a)/float(NDIV)
        integrate=0.0;
        do i=1,NDIV
          x1= a + float(i-1)*dx;
          x2= a + float( i )*dx;
          integrate =integrate + qsimp(x1,x2,es,kappa,eta,gama,C1,tipo);
        end do
        return
      end function integrate
C ------------------------------------------------------------------------------
      function qsimp(x1,x2,es,kappa,eta,gama,C1,tipo)
C ------------------------------------------------------------------------------
C Adaptive simpson rule
C ------------------------------------------------------------------------------
        INCLUDE 'aba_param.inc'

        real*8,parameter   :: Ct= 1.0/3.0, EP1=3.0e-7, EP2=1.0e-8
        integer, parameter :: JMAX = 12
        real*8, intent(in) :: x1,x2,es,kappa,eta,gama,C1
        integer, intent(in):: tipo
        real*8             :: qsimp,st,s,os,ost
        integer            :: j

        ost = 0.0; os = 0.0;
        do j=1,JMAX
          st=trapzd(ost,x1,x2,es,kappa,eta,gama,C1,j,tipo)
          s=(4.0*st-ost)*Ct;
          if (j > 5) then
            if((abs(s-os)<EP1*abs(os)).or.((abs(s)<=EP2).and.
	1	     (abs(os)<=EP2))) then
              qsimp=s
              return;
            end if
          end if
          os=s; ost=st;
        end do
        write(*,*) 'Too many steps in routine qsimp'
        return
      end function qsimp
C ------------------------------------------------------------------------------
      function trapzd(trapz_old,x1,x2,es,kappa,eta,gama,C1,n,tipo)
C ------------------------------------------------------------------------------
C Trapezoidal integration rule
C ------------------------------------------------------------------------------
        INCLUDE 'aba_param.inc'

        real*8, intent(in) :: trapz_old,x1,x2,es,kappa,eta,gama,C1
        integer, intent(in):: n,tipo
        integer            :: j,it
        real*8             :: trapzd,Fa,Fb,tnm,del,x,suma

       if (n == 1) then
         Fa=integrand(x1,es,kappa,eta,gama,C1,tipo);
         Fb=integrand(x2,es,kappa,eta,gama,C1,tipo);
         trapzd=0.5*(x2-x1)*(Fa+Fb);
       else
         it=2.0**(n-2);
         tnm=1.0/float(it)
         del=(x2-x1)*tnm
         x=x1+0.5*del;
         suma=0.0;
         do j=1,it
           x=x+del;
           suma = suma+integrand(x,es,kappa,eta,gama,C1,tipo);
         end do
         trapzd=0.5*(trapz_old+del*suma);
       end if
       return
      end function trapzd
C ------------------------------------------------------------------------------
      function integrand(x,es,kappa,eta,gama,D1,tipo)
C ------------------------------------------------------------------------------
C Integrand for Appell hypergeometric function
C
C x       ... point of evaluation
C es      ... strain level
C kappa   ... damage parameter
C eta     ... damage parameter
C gama    ... damage parameter
C D1      ... damage parameter
C tipo    ... Type of integrand for first or second derivative
C ------------------------------------------------------------------------------
        INCLUDE 'aba_param.inc'

        integer,intent(in) :: tipo
        real*8, intent(in) :: x,es,kappa,eta,gama,D1
        real*8             :: term1,a,b,term2,term3,integrand

        if (tipo==1) then
          if(abs(es) < 1e-8) then
            term1=1.0;
          else
            term1=x*x/((x-es)*(x-es));
          end if
          a=x/kappa;
          b=gama-1.0;
          term2=a**b;
          a=1-x/kappa;
          b=eta-1;
          term3=a**b;
          integrand = (D1/kappa)*term1*term2*term3;
        elseif(tipo==2) then
          if(abs(es)<1e-8) then
            term1=1.0;
          else
            term1=x*x*x/((x-es)*(x-es)*(x-es));
          end if
          a=x/kappa;
          b=gama-2.0;
          term2=a**b;
          a=1-x/kappa;
          b=eta-1;
          term3=a**b;
          integrand = 2.0*(D1/(kappa*kappa))*term1*term2*term3;
        end if
        return
      end function integrand
C ------------------------------------------------------------------------------
      subroutine deftensors(FG,na0,nb0,tenb,tenb2,inva)
C ------------------------------------------------------------------------------
C Calculates invariants and deformation tensors
C ------------------------------------------------------------------------------
        INCLUDE 'aba_param.inc'

        real*8, intent(in)  :: FG(3,3),na0(3,1),nb0(3,1)
        real*8, intent(out) :: tenb(6,1)
        real*8, intent(out) :: tenb2(6,1),inva(4)
        real*8              :: tenb1(3,3), tenb12(3,3)
        real*8              :: tenC(3,3), vtemp(3,1)
        real*8              :: traceb2
        integer             :: i

        tenC=matmul(transpose(FG),FG)
        tenb1=matmul(FG,transpose(FG))
        tenb12=matmul(tenb1,tenb1);
C
C Invariants
C
        inva(1)=sum((/(tenb1(i,i),i=1,3)/))
        traceb2=sum((/(tenb12(i,i),i=1,3)/))
        inva(2)=0.5*(inva(1)*inva(1)-traceb2)
        vtemp=matmul(tenC,na0)
        inva(3)=dot_product(na0(:,1),vtemp(:,1));
        vtemp=matmul(tenC,nb0)
        inva(4)=dot_product(nb0(:,1),vtemp(:,1));
C
C Deformation tensors
C
       tenb(1:3,1)=(/(tenb1(i,i),i=1,3)/)
       tenb(4:6,1)=(/tenb1(1,2),tenb1(1,3),tenb1(2,3)/)
       tenb2(1:3,1)=(/(tenb12(i,i),i=1,3)/)
       tenb2(4:6,1)=(/tenb12(1,2),tenb12(1,3),tenb12(2,3)/)
       end subroutine deftensors
C
C ------------------------------------------------------------------------------
      subroutine sigma (n_te,p,dw1,inva,na,nb,tenb,tenb2,jac,
     1	              Sv,Si,Sa1,Sa2)
C ------------------------------------------------------------------------------
C Function to calculate the cauchy stress tensor
C
C Vectors and tensors are organized as follows
C
C dw1=[W1, W2, W4, W6];
C inva=[I1, I2, I4, I6];
C tenb=[b11 b22 b33 b12 b13 b23]'; (same order for tenb2)
C na=[a1 a2 a3]';
C nb=[b1 b2 b3]';
C ------------------------------------------------------------------------------
        INCLUDE 'aba_param.inc'

        integer, intent(in):: n_te
        real*8, intent(in) :: p,dw1(4),inva(4),na(3,1),nb(3,1)
        real*8, intent(in) :: tenb(n_te,1),tenb2(n_te,1),jac
        real*8, intent(out):: Sv(n_te,1),Si(n_te,1)
	  real*8, intent(out):: Sa1(n_te,1),Sa2(n_te,1)
        real*8             :: coef,Iso1,Iso2,Iso3,Aiso1,Aiso2
        real*8             :: uno(n_te,1),tA(n_te,1),tB(n_te,1)

        coef=2.0/(3.0*jac)
        uno=0.0
        uno(1:3,1)=1.0
C volumetric component
        Sv=p*uno
C Isotropic component
        Iso1=(-coef)*(dw1(1)*inva(1)+2.0*dw1(2)*inva(2))
        Iso2=  3.0*coef*(dw1(1)+inva(1)*dw1(2))
        Iso3=(-3.0*coef)*dw1(2)
        Si=Iso1*uno+Iso2*tenb+Iso3*tenb2
C AnIsotropic component
C Familia 1
        tA(1:3,1)=(/na(1,1)*na(1,1),na(2,1)*na(2,1),na(3,1)*na(3,1)/)
        tA(4:6,1)=(/na(1,1)*na(2,1),na(1,1)*na(3,1),na(2,1)*na(3,1)/)
	  Aiso1=(-coef)*(dw1(3)*inva(3))
	  Sa1=Aiso1*uno+3*coef*(dw1(3)*inva(3)*tA)
C Familia 2
        tB(1:3,1)=(/nb(1,1)*nb(1,1),nb(2,1)*nb(2,1),nb(3,1)*nb(3,1)/)
        tB(4:6,1)=(/nb(1,1)*nb(2,1),nb(1,1)*nb(3,1),nb(2,1)*nb(3,1)/)
        Aiso2=(-coef)*(dw1(4)*inva(4))
	 
        Sa2=Aiso2*uno+3*coef*(dw1(4)*inva(4)*tB)
      end subroutine sigma
C ------------------------------------------------------------------------------
C
C Routines to calculate the elasticity tensors
C
C ------------------------------------------------------------------------------
      subroutine Cvol(n_te,pbar1,pbar2,Cv)
C ------------------------------------------------------------------------------
C Function to calculate the volumetric term
C of the elasticity tensor
C
C Vectors and tensors are organized as follows
C
C dw1=[W1, W2, W4, W6];
C dw2=[W11, W22, W44, W66, W12, W14, W16, W24, W26, W46];
C inv=[I1, I2, I4, I6];
C tenb=[b11 b22 b33 b12 b13 b23]'; (same order for tenb2)
C na=[a1 a2 a3]';
C nb=[b1 b2 b3]';
C ------------------------------------------------------------------------------

        include "aba_param.inc"
        integer, intent(in):: n_te
        real*8, intent(in) :: pbar1,pbar2
        real*8,intent(out) :: Cv(n_te,n_te)
        real*8             :: uno(n_te,n_te),Id(n_te,n_te)
        dimension          :: i(3)
        data i /1,2,3/

        uno=0.0
        uno(i,i)=1.0
        call eye(6,Id)
        Id(4,4)=0.5; Id(5,5)=0.5; Id(6,6)=0.5;

        Cv=pbar2*uno-2.0*pbar1*Id
      end subroutine Cvol
C ------------------------------------------------------------------------------
      subroutine Ciso(n_te,dw1,dw2,inva,tenb,tenb2,jac,Ci)
C ------------------------------------------------------------------------------
C Function to calculate the isotropic term
C of the deviatoric componenet of
C the elasticity tensor
C
C Vectors and tensors are organized as follows
C
C dw1=[W1, W2, W4, W6];
C dw2=[W11, W22, W44, W66, W12, W14, W16, W24, W26, W46];
C inva=[I1, I2, I4, I6];
C tenb=[b11 b22 b33 b12 b13 b23]'; (same order for tenb2)
C na=[a1 a2 a3]';
C nb=[b1 b2 b3]';

        integer, intent(in) :: n_te
        real*8, intent(in)  :: dw1(4),dw2(10),inva(4),tenb(n_te,1)
        real*8, intent(in)  :: tenb2(n_te,1),jac
        real*8, intent(out) :: Ci(n_te,n_te)
        real*8              :: coef1,coef2,coef3
        real*8              :: Iso1,Iso2,Iso3,Iso4,Iso5,Iso6
        real*8              :: uno(n_te,1),Id(n_te,n_te),b1(n_te,n_te)

        coef3=4.0/(9.0*jac)
        coef1=3.0*coef3
        coef2=9.0*coef3

        Iso1=inva(1)*(2.0*dw1(2)+dw2(1)+inva(1)*dw2(5))
        Iso1=Iso1+dw1(1)+2*inva(2)*(inva(1)*dw2(2)+dw2(5))
        Iso1=(-coef1)*Iso1
        Iso2=2.0*dw1(2)+inva(1)*dw2(5)+2*inva(2)*dw2(2)
        Iso2=coef1*Iso2
        Iso3=dw2(1)+dw1(2)+inva(1)*(2*dw2(5)+inva(1)*dw2(2))
        Iso3=coef2*Iso3
        Iso4=dw2(5)+inva(1)*dw2(2)
        Iso4=(-coef2)*Iso4
        Iso5=dw1(1)*inva(1)+2*inva(2)*dw1(2)
        Iso5=coef1*Iso5
        Iso6=inva(1)*(dw1(1)+inva(1)*dw2(1))
        Iso6=Iso6+4*inva(2)*(dw1(2)+inva(1)*dw2(5)+inva(2)*dw2(2))
        Iso6=coef3*Iso6;

        uno=0.0
        uno(1:3,1)=1.0
        call eye(n_te,Id)
        Id(4,4)=0.5
        Id(5,5)=0.5
        Id(6,6)=0.5

        Ci=0.0
        b1=matmul(tenb,transpose(uno))
        Ci=Ci+Iso1*(b1+transpose(b1))
        b1=matmul(tenb2,transpose(uno))
        Ci=Ci+Iso2*(b1+transpose(b1))
        b1=matmul(tenb2,transpose(tenb2))
        Ci=Ci+coef2*dw2(2)*b1
        b1=matmul(tenb,transpose(tenb))
        Ci=Ci+Iso3*b1;
        b1=matmul(tenb2,transpose(tenb))
        Ci=Ci+Iso4*(b1+transpose(b1));
        Ci=Ci+Iso5*Id;
        b1=matmul(uno,transpose(uno))
        Ci=Ci+Iso6*b1;
C
C tensor Ib
C
        b1=0.0
C
C first quadrant
C
        b1(1,1)=0.5*tenb(1,1)*tenb(1,1);
        b1(2,2)=0.5*tenb(2,1)*tenb(2,1);
        b1(3,3)=0.5*tenb(3,1)*tenb(3,1);
        b1(1,2)=tenb(4,1)*tenb(4,1);
        b1(1,3)=tenb(5,1)*tenb(5,1);
        b1(2,3)=tenb(6,1)*tenb(6,1);
C
C second quadrant
C
        b1(1,4)=tenb(1,1)*tenb(4,1);
        b1(1,5)=tenb(1,1)*tenb(5,1);
        b1(1,6)=tenb(4,1)*tenb(5,1);
        b1(2,4)=tenb(2,1)*tenb(4,1);
        b1(2,5)=tenb(4,1)*tenb(6,1);
        b1(2,6)=tenb(2,1)*tenb(6,1);
        b1(3,4)=tenb(5,1)*tenb(6,1);
        b1(3,5)=tenb(3,1)*tenb(5,1);
        b1(3,6)=tenb(3,1)*tenb(6,1);
C
C fourth quadrant
C
       b1(4,4)=0.25*(tenb(1,1)*tenb(2,1)+tenb(4,1)*tenb(4,1));
       b1(5,5)=0.25*(tenb(1,1)*tenb(3,1)+tenb(5,1)*tenb(5,1));
       b1(6,6)=0.25*(tenb(2,1)*tenb(3,1)+tenb(6,1)*tenb(6,1));
       b1(4,5)=0.5*(tenb(1,1)*tenb(6,1)+tenb(4,1)*tenb(5,1));
       b1(4,6)=0.5*(tenb(4,1)*tenb(6,1)+tenb(2,1)*tenb(5,1));
       b1(5,6)=0.5*(tenb(3,1)*tenb(4,1)+tenb(5,1)*tenb(6,1));
C
C Tensor Ci
C
       Ci=Ci-coef2*dw1(2)*(b1+transpose(b1));
      end subroutine Ciso
C ------------------------------------------------------------------------------
      subroutine Caniso2(n_te,dw1,dw2,inva,na,nb,tenb,tenb2,jac,
	1                  Ca1,Ca2)
C ------------------------------------------------------------------------------
C Function to calculate the anisotropic term
C of the deviatoric component of
C the elasticity tensor
C
C Vectors and tensors are organized as follows
C
C dw1=[W1, W2, W4, W6];
C dw2=[W11, W22, W44, W66, W12, W14, W16, W24, W26, W46];
C inva=[I1, I2, I4, I6];
C tenb=[b11 b22 b33 b12 b13 b23]'; (same order for tenb2)
C na=[a1 a2 a3]';
C nb=[b1 b2 b3]';

        integer, intent(in):: n_te
        real*8, intent(in) :: dw1(4),dw2(10),inva(4),na(3,1),nb(3,1)
        real*8, intent(in) :: tenb(n_te,1),tenb2(n_te,1),jac
        real*8, intent(out):: Ca1(n_te,n_te), Ca2(n_te,n_te)
        real*8             :: coef1,coef2,coef3
        real*8             :: Aiso1a,Aiso2a,Aiso3a,Aiso4a
        real*8             :: Aiso5,Aiso6,Aiso7,Aiso8
	  real*8             :: Aiso1b,Aiso2b,Aiso3b,Aiso4b
      
        real*8             :: uno(1,n_te),Id(n_te,n_te)
        real*8             :: tA(n_te,1),tB(n_te,1),b1(n_te,n_te)

        coef2=4.0/(9.0*jac)
        coef1=3.0*coef2
        coef3=9.0*coef2
C
        Aiso1a=inva(3)*dw2(6) + inva(1)*(inva(3)*dw2(8))
        Aiso1b=inva(4)*dw2(7) + inva(1)*(inva(4)*dw2(9))
        Aiso1a=(-coef1)*Aiso1a
	  Aiso1b=(-coef1)*Aiso1b
C
        Aiso2a=coef1*(inva(3)*dw2(8))
	  Aiso2b=coef1*(inva(4)*dw2(9))
C
C No considero los terminos cruzados       
C        Aiso3=2*inva(1)*(inva(3)*dw2(6)+inva(4)*dw2(7))
C       Aiso3=Aiso3+4*inva(2)*(inva(3)*dw2(8)+inva(4)*dw2(9))
C        Aiso3=Aiso3+inva(3)*(inva(3)*dw2(3)+dw1(3))
C        Aiso3=Aiso3+inva(4)*(inva(4)*dw2(4)+dw1(4))
C        Aiso3=Aiso3+2*inva(3)*inva(4)*dw2(10)
C        Aiso3=coef2*Aiso3


        Aiso3a=coef2*(inva(3)*(inva(3)*dw2(3)+dw1(3)))
        Aiso3b=coef2*(inva(4)*(inva(4)*dw2(4)+dw1(4)))



        Aiso4a=coef1*inva(3)*dw1(3)
	  Aiso4b=coef1*inva(4)*dw1(4)
      
C De moemento consodero que w46=0, y que todo es de la fibra familia a
        Aiso5=inva(3)*(dw1(3)+inva(1)*dw2(6)+2*inva(2)*dw2(8))
        Aiso5=Aiso5+inva(3)*(inva(4)*dw2(10)+inva(3)*dw2(3))
        Aiso5=(-coef1)*Aiso5
C De moemento consodero que w46=0, y que todo es de la fibra familia b

        Aiso6=inva(4)*(dw1(4)+inva(1)*dw2(7)+2*inva(2)*dw2(9))
        Aiso6=Aiso6+inva(4)*(inva(3)*dw2(10)+inva(4)*dw2(4))
        Aiso6=(-coef1)*Aiso6
C Familia a
        Aiso7=inva(3)*(dw2(6)+inva(1)*dw2(8))
        Aiso7=coef3*Aiso7
C Familia b
        Aiso8=inva(4)*(dw2(7)+inva(1)*dw2(9))
        Aiso8=coef3*Aiso8

        uno=0.0
        uno(1,1:3)=1.0
        call eye(n_te,Id)
        Id(4,4)=0.5
        Id(5,5)=0.5
        Id(6,6)=0.5
        tA(1:3,1)=(/na(1,1)*na(1,1),na(2,1)*na(2,1),na(3,1)*na(3,1)/)
        tA(4:6,1)=(/na(1,1)*na(2,1),na(1,1)*na(3,1),na(2,1)*na(3,1)/)
        tB(1:3,1)=(/nb(1,1)*nb(1,1),nb(2,1)*nb(2,1),nb(3,1)*nb(3,1)/)
        tB(4:6,1)=(/nb(1,1)*nb(2,1),nb(1,1)*nb(3,1),nb(2,1)*nb(3,1)/)

        Ca1=0.0
	  Ca2=0.0

        b1=matmul(tenb,uno)
        Ca1=Ca1+Aiso1a*(b1+transpose(b1))
	  Ca2=Ca2+Aiso1b*(b1+transpose(b1))

        b1=matmul(tenb2,uno)
        Ca1=Ca1+Aiso2a*(b1+transpose(b1))
        Ca2=Ca2+Aiso2b*(b1+transpose(b1))

        b1=matmul(transpose(uno),uno)
        Ca1=Ca1+Aiso3a*b1
	  Ca2=Ca2+Aiso3b*b1

        Ca1=Ca1+Aiso4a*Id
	  Ca2=Ca2+Aiso4b*Id

        b1=matmul(tA,uno)
        Ca1=Ca1+Aiso5*(b1+transpose(b1))
C
        b1=matmul(tB,uno)
        Ca2=Ca2+Aiso6*(b1+transpose(b1))
C
        b1=matmul(tA,transpose(tenb))
        Ca1=Ca1+Aiso7*(b1+transpose(b1))
C
        b1=matmul(tB,transpose(tenb))
        Ca2=Ca2+Aiso8*(b1+transpose(b1))
C
        b1=matmul(tA,transpose(tenb2))
        Ca1=Ca1-coef3*dw2(8)*inva(3)*(b1+transpose(b1))
C
        b1=matmul(tB,transpose(tenb2))
        Ca2=Ca2-coef3*dw2(9)*inva(4)*(b1+transpose(b1))
C
C   No considero acoplamiento entre familias
C        b1=matmul(tB,transpose(tA))
C       Ca=Ca+coef3*inva(3)*inva(4)*dw2(10)*(b1+transpose(b1))
C
        b1=matmul(tA,transpose(tA))
        Ca1=Ca1+coef3*inva(3)*inva(3)*dw2(3)*b1
        b1=matmul(tB,transpose(tB))
        Ca2=Ca2+coef3*inva(4)*inva(4)*dw2(4)*b1;
      end subroutine Caniso2

C ------------------------------------------------------------------------------
      subroutine Caniso(n_te,dw1,dw2,inva,na,nb,tenb,tenb2,jac,
	1                  Ca)
C ------------------------------------------------------------------------------
C 
C Calcula de forma independiente la contribucion de cada familia de fibras
C Function to calculate the anisotropic term
C of the deviatoric component of
C the elasticity tensor
C
C Vectors and tensors are organized as follows
C
C dw1=[W1, W2, W4, W6];
C dw2=[W11, W22, W44, W66, W12, W14, W16, W24, W26, W46];
C inva=[I1, I2, I4, I6];
C tenb=[b11 b22 b33 b12 b13 b23]'; (same order for tenb2)
C na=[a1 a2 a3]';
C nb=[b1 b2 b3]';

        integer, intent(in):: n_te
        real*8, intent(in) :: dw1(4),dw2(10),inva(4),na(3,1),nb(3,1)
        real*8, intent(in) :: tenb(n_te,1),tenb2(n_te,1),jac
        real*8, intent(out):: Ca(n_te,n_te)
        real*8             :: coef1,coef2,coef3
        real*8             :: Aiso1,Aiso2,Aiso3,Aiso4
        real*8             :: Aiso5,Aiso6,Aiso7,Aiso8
        real*8             :: uno(1,n_te),Id(n_te,n_te)
        real*8             :: tA(n_te,1),tB(n_te,1),b1(n_te,n_te)

        coef2=4.0/(9.0*jac)
        coef1=3.0*coef2
        coef3=9.0*coef2
        Aiso1=inva(3)*dw2(6)+inva(4)*dw2(7)
        Aiso1=Aiso1+inva(1)*(inva(3)*dw2(8)+inva(4)*dw2(9))
        Aiso1=(-coef1)*Aiso1
        Aiso2=inva(3)*dw2(8)*inva(4)*dw2(9)
        Aiso2=coef1*Aiso2
        Aiso3=2*inva(1)*(inva(3)*dw2(6)+inva(4)*dw2(7))
        Aiso3=Aiso3+4*inva(2)*(inva(3)*dw2(8)+inva(4)*dw2(9))
        Aiso3=Aiso3+inva(3)*(inva(3)*dw2(3)+dw1(3))
        Aiso3=Aiso3+inva(4)*(inva(4)*dw2(4)+dw1(4))
        Aiso3=Aiso3+2*inva(3)*inva(4)*dw2(10)
        Aiso3=coef2*Aiso3
        Aiso4=inva(3)*dw1(3)+inva(4)*dw1(4)
        Aiso4=coef1*Aiso4
        Aiso5=inva(3)*(dw1(3)+inva(1)*dw2(6)+2*inva(2)*dw2(8))
        Aiso5=Aiso5+inva(3)*(inva(4)*dw2(10)+inva(3)*dw2(3))
        Aiso5=(-coef1)*Aiso5
        Aiso6=inva(4)*(dw1(4)+inva(1)*dw2(7)+2*inva(2)*dw2(9))
        Aiso6=Aiso6+inva(4)*(inva(3)*dw2(10)+inva(4)*dw2(4))
        Aiso6=(-coef1)*Aiso6
        Aiso7=inva(3)*(dw2(6)+inva(1)*dw2(8))
        Aiso7=coef3*Aiso7
        Aiso8=inva(4)*(dw2(7)+inva(1)*dw2(9))
        Aiso8=coef3*Aiso8

        uno=0.0
        uno(1,1:3)=1.0
        call eye(n_te,Id)
        Id(4,4)=0.5
        Id(5,5)=0.5
        Id(6,6)=0.5
        tA(1:3,1)=(/na(1,1)*na(1,1),na(2,1)*na(2,1),na(3,1)*na(3,1)/)
        tA(4:6,1)=(/na(1,1)*na(2,1),na(1,1)*na(3,1),na(2,1)*na(3,1)/)
        tB(1:3,1)=(/nb(1,1)*nb(1,1),nb(2,1)*nb(2,1),nb(3,1)*nb(3,1)/)
        tB(4:6,1)=(/nb(1,1)*nb(2,1),nb(1,1)*nb(3,1),nb(2,1)*nb(3,1)/)

        Ca=0.0

        b1=matmul(tenb,uno)
        Ca=Ca+Aiso1*(b1+transpose(b1))
        b1=matmul(tenb2,uno)
        Ca=Ca+Aiso2*(b1+transpose(b1))
        b1=matmul(transpose(uno),uno)
        Ca=Ca+Aiso3*b1
        Ca=Ca+Aiso4*Id
        b1=matmul(tA,uno)
        Ca=Ca+Aiso5*(b1+transpose(b1))
        b1=matmul(tB,uno)
        Ca=Ca+Aiso6*(b1+transpose(b1))
        b1=matmul(tA,transpose(tenb))
        Ca=Ca+Aiso7*(b1+transpose(b1))
        b1=matmul(tB,transpose(tenb))
        Ca=Ca+Aiso8*(b1+transpose(b1))
        b1=matmul(tA,transpose(tenb2))
        Ca=Ca-coef3*dw2(8)*inva(3)*(b1+transpose(b1))
        b1=matmul(tB,transpose(tenb2))
        Ca=Ca-coef3*dw2(9)*inva(4)*(b1+transpose(b1))
        b1=matmul(tB,transpose(tA))
        Ca=Ca+coef3*inva(3)*inva(4)*dw2(10)*(b1+transpose(b1))
        b1=matmul(tA,transpose(tA))
        Ca=Ca+coef3*inva(3)*inva(3)*dw2(3)*b1
        b1=matmul(tB,transpose(tB))
        Ca=Ca+coef3*inva(4)*inva(4)*dw2(4)*b1;
      end subroutine Caniso
C
C  ------------------------------------------------------------------------------
      subroutine eye(n,Id)
C ------------------------------------------------------------------------------
C Returns a nxn identity matrix
C
        INCLUDE 'aba_param.inc'

        integer, intent(in):: n
        real*8,intent(out) :: Id(n,n)
        integer            :: i

        Id=0.0
        do i=1,n
           Id(i,i)=1.0
        end do
        return
        end subroutine eye
C
C ------------------------------------------------------------------------------
        subroutine det(A,jac)
C ------------------------------------------------------------------------------
C Returns the determinant of A
C
        INCLUDE 'aba_param.inc'

        real*8, intent(in) :: A(3,3)
        real*8, intent(out):: jac
        real*8             :: x(3)
        integer            :: i(3),j(3)
        data i /2,3,1/; data j /3,1,2/

         x=A(2,i)*A(3,j)-A(2,j)*A(3,i)
         jac=sum(A(1,:)*x)
        end subroutine det
C
C ------------------------------------------------------------------------------
        function gammaln(x)
C ------------------------------------------------------------------------------
C Calculates the logarithm of the gamma function
C
        INCLUDE 'aba_param.inc'

         real*8, intent(in)  :: x
         real*8              :: gammaln
         real*8              :: tmp,ser,cof(6),prog(6)
         integer             :: j

         cof=(/76.18009173,-86.50532033,24.01409822,
     1     -1.231739516,0.120858003e-2,-0.536382e-5/)

         if(x<0) then
           gammaln=1.0
         else
           tmp=x+5.5
           tmp=(x+0.5)*log(tmp)-tmp
           prog=(/(1.0/(x+j),j=1,size(cof))/)
           ser=2.5066282746310005*(1.000000000190015+sum(cof*prog))/x
           gammaln=tmp+log(ser)
         end if
        end function gammaln
C ------------------------------------------------------------------------------
      subroutine inv(A,invA)
C ------------------------------------------------------------------------------
C Returns the inverse of A. A 3x3 matrix
C ------------------------------------------------------------------------------

      implicit none
      real*8,intent(in) :: A(3,3)
      real*8,intent(out):: invA(3,3)
      real*8            :: detA

		call det(A,detA)
		if (abs(detA).LT.1e-6) then
			 write(*,*) 'Singular Matrix in Function inv(.)'
		else
		  detA=1.0/detA
		  invA(1,1)=(A(2,2)*A(3,3)-A(2,3)*A(3,2))
		  invA(1,2)=-(A(1,2)*A(3,3)-A(1,3)*A(3,2))
		  invA(1,3)=(A(1,2)*A(2,3)-A(1,3)*A(2,2))
		  invA(2,1)=-(A(2,1)*A(3,3)-A(2,3)*A(3,1))
		  invA(2,2)=(A(1,1)*A(3,3)-A(1,3)*A(3,1))
		  invA(2,3)=-(A(1,1)*A(2,3)-A(1,3)*A(2,1))
		  invA(3,1)=(A(2,1)*A(3,2)-A(2,2)*A(3,1))
		  invA(3,2)=-(A(1,1)*A(3,2)-A(1,2)*A(3,1))
		  invA(3,3)=(A(1,1)*A(2,2)-A(1,2)*A(2,1))
		  invA=invA*detA
		endif
      return
      end subroutine inv
C
C ------------------------------------------------------------------------------
      subroutine  b_search (arr,n,key,pos,fou)
C ------------------------------------------------------------------------------
C busqueda binaria
C datos:
C       arr,ind: array de datos e indices
C       n    : dimension maxima del array a e ind
C       key  : elemento buscado
C       pos  : posicion del elemento en el array
C       fou  : variable logica, cierta se se ha encontrado el elemento
C ------------------------------------------------------------------------------
      implicit none
      integer, dimension (:),intent (in) :: arr
      integer, intent (in)               :: key
      integer, intent (in)               :: n
      integer, intent (out)              :: pos
      logical, intent (out)              :: fou
      integer                            :: l,r,mid,k

		fou =.false.
		l = lbound(arr,1)
		if (n < l) then
		  return
		end  if
		r = n
		if (key <= n) then
		  if (key == arr(key)) then
			pos=key; fou=.true.
			return
		  elseif (key > arr(key)) then
			l = key
		  else
			r = key
		  end if
		end if
C
		do
		   mid = (l+r)/2
		   k   = mid
		   if (key < arr(k))then
			  if (l >= r) then
				 exit
			  end  if
			  r = mid
		   elseif (key > arr(k)) then
			  if (l >= r) then
				 exit
			  end  if
			  l = mid+1
		   else
			  pos=  k
			  fou= .true.
			  exit
		   end  if
		end  do
		return
	 end subroutine
C
C ------------------------------------------------------------------------------
C dmg_matriz
C Calcula el da�o en la matriz para el modelo CDM de Simo 1987c
C IMPORTANTE: Antes de utilizar la rutinas actualida la funcion da�o
C			que desee utilizar (var tipodam)
C wengy(1)... total strain energy
C wengy(2)... matrix strain energy
C damM(1) ... damage strain for damage evaluation for matrix
C damM(2) ... Damage variable for matrix
C ddam	... Derivative of damage function for matrix
C ------------------------------------------------------------------------------
 	subroutine dmg_matriz(PROPS,NPROPS,wengy,damM,ddam)
C
	dimension PROPS(NPROPS),damM(6),WENGY(4)
	double precision PROPS, damM
	double precision WENGY,ddam
      double precision double  bb, linm,lfim,betam,etrial
	integer tipodam

C Recuperamos los parametros del modelo de da�o para la matriz solidad*/
        linm = PROPS(18)
        lfim = PROPS(19)
        betam = PROPS(20)

C Definimos la funcion de da�o que queremos: 
C	tipodam=1 (Simo), tipodam=2 (cubica), tipodam=3 (sigmoide), tipodam=4 (sigmoide mod)
	 tipodam= PROPS(27) 

        etrial=sqrt(2*wengy(2))
        ddam=0.0
	  
	  bb= (etrial-linm)/(lfim-linm)

C   Calcula los parametros de dmg: la variable trial y si hay o no dmg */

	  if(( etrial > damM(1))) then
             damM(1)=etrial
		if((etrial > linm) .AND. (etrial < lfim)) then
			if (tipodam == 1) then
				bb= exp(betam * (linm-lfim))
				aa= dexp(betam * (etrial-lfim))
				damM(2) = 1.0 - ((1.0 - aa)/(1.0 -bb))
				ddam = betam * aa /bb 
			else if (tipodam == 2) then
				damM(2) = bb**(2)*(1-betam*(bb**(2)-1))
				ddam = 4*betam*bb**(3) - 2*bb*(1+betam)
			else if (tipodam == 3) then
				expon=2.0*linm*(2.0*etrial/lfim-1.0)
				ter1=exp(expon)
				ter2=etrial*linm*exp(expon)-1.0
				ter3=1.0/(2+ter2)
				damM(4)=0.5*(1.0+ter2*ter3)
				ddam=2.0*(1-damM(4))*(linm/lfim)*
	1				(4.0*etrial*linm+lfim)*ter1*ter3
			else 
				expon=2.0*betam*(2.0*bb-1.0)
				ter1=exp(expon)
				ter2=2.0*betam*bb*ter1+1.0
				ter3=2.0*betam*bb*ter1-1.0
				damM(4)=0.5*(1.0+ter2/ter3)
				ddam=betam*ter1*(betam*bb*(betam*ter1*(bb*(8.0-
	1				16.0*betam*ter1)-8.0)-4.0)-2.0)/ter2**(2)
			end if
		else if (etrial > lfim) then 
		    damM(2) = 1.0  
			ddam = 0.0
		end if
   	  end if

      return
	end subroutine dmg_matriz
C
C ------------------------------------------------------------------------------
C dmg_continuom
C Calcula el da�o continuo y discontinuo en la matriz segun el articulo de Miehe 1997
C IMPORTANTE: Antes de utilizar la rutinas actualida la funcion da�o
C			que desee utilizar (var tipodam)
C wengy(1)... total strain energy
C wengy(2)... matrix strain energy
C damM(1) ... damage strain for damage evaluation for matrix
C damM(2) ... Damage variable for matrix
C ddam	... Derivative of damage function for matrix

C damM(1)=maxetrial_m,damM(2)=dam_m,damM(3)=maxetrial_m_disco,damM(4)=dam_m_disco,
C damF(5)=etrial_m_cont,damM(6)=dam_m_cont, 

C ------------------------------------------------------------------------------
c
 	subroutine DMGcont_matriz(PROPS,NPROPS,wengy,damM,ddam)
C
	dimension PROPS(NPROPS),damM(6),wengy(4)
	double precision PROPS, damM
	double precision wengy,ddam,ddama,ddamb
	double precision etrial, dbinfinito, nubeta, beta, aa
	double precision linm, lfim, betam
	integer tipodam

C Definimos la funcion de da�o que queremos: 
C	tipodam=1 (Simo), tipodam=2 (cubica), tipodam=3 (sigmoide), tipodam=4 (sigmoide mod)
C Hay que definirlo en el fichero porque no hay espacio para constantes
	 tipodam=4

C Recuperamos los parametros del modelo de da�o para la matriz solida*/
	  linm = PROPS(18)
        lfim = PROPS(19)
        betam = PROPS(20)
        dbinfinito = PROPS(21)
	  nubeta = PROPS(22)

	  ddam=0.0; ddama=0.0; ddamb=0.0; beta=0.0;
        etrial=sqrt(2*wengy(2))	  
	  bb= (etrial-linm)/(lfim-linm)
               
C Calculamos primero el damg discontinuo*/

	  if(( etrial > damM(3))) then  
	    damM(3)=etrial
	    if((etrial > linm) .AND. (etrial < lfim)) then
			if (tipodam == 1) then
				bb= exp(betam * (linm-lfim))
				aa= dexp(betam * (etrial-lfim))
				damM(4) = 1.0 - ((1.0 - aa)/(1.0 -bb))
				ddama = betam * aa /bb 
			else if (tipodam == 2) then
		        damM(4) = bb**(2)*(1-betam*(bb**(2)-1))
		        ddama = 4*betam*bb**(3) - 2*bb*(1+betam)
			else if (tipodam == 3) then
			    expon=2.0*linm*(2.0*etrial/lfim-1.0)
			    ter1=exp(expon)
			    ter2=etrial*linm*exp(expon)-1.0
			    ter3=1.0/(2+ter2)
			    damM(4)=0.5*(1.0+ter2*ter3)
			    ddama=2.0*(1-damM(4))*(linm/lfim)*
	1		    (4.0*etrial*linm+lfim)*ter1*ter3
			else
				expon=2.0*betam*(2.0*bb-1.0)
				ter1=exp(expon)
				ter2=2.0*betam*bb*ter1+1.0
				ter3=2.0*betam*bb*ter1-1.0
				damM(4)=0.5*(1.0+ter2/ter3)
				ddama=betam*ter1*(betam*bb*(betam*ter1*(bb*(8.0-
	1				16.0*betam*ter1)-8.0)-4.0)-2.0)/ter2**(2)
			end if
	    else if (etrial > lfim) then 
		    damM(4) = 1.0  
			ddama = 0.0
		end if
   	  end if
               
C Calculamos ahora el damg continuo*/
	 if(abs(PROPS(22)) <= 1.0E-6) then
	     damM(6) = 0.0
		 ddamb = 0.0
	 else
		 beta = damM(5) + abs(etrial - damM(1))      
		 damM(6) = dbinfinito*(1-dexp(-(beta/nubeta)))
		 if(etrial> damM(1)) then
			ddamb = dbinfinito*dexp(-(beta/nubeta))
		 else
			ddamb = -1.0*dbinfinito*dexp(-(beta/nubeta))
		 end if
		 damM(5)= beta
		 damM(1)= etrial
   	  end if

	  damM(2) =  damM(4) + damM(6)
	  ddam = ddama + ddamb
	  if(damM(2)>= 1) then
		damM(2)= 1.0
		ddam= 0.0
	  end if     
        
      return
	end subroutine DMGcont_matriz
C
C ------------------------------------------------------------------------------
C DMGpseudo_matriz
C Calcula el da�o en la matriz segun el modelo pseudoelastico de Ogden&Roxburg 1999 y modificado por Bose
C IMPORTANTE: Para poder utilizar el esquema general del codigo para el da�o (tensiones y tensor elastico)
C			se envia al principal 1-eta y -damM
C ------------------------------------------------------------------------------
 	subroutine DMGpseudo_matriz(PROPS,NPROPS,wengy,damM,ddamM)
C
	dimension PROPS(NPROPS),damM(6),wengy(4)
	double precision PROPS, damM, wengy,ddamM

      double precision alfam,betam,gammam,erfm,aa,erfp,bb,etam,Pi,var

C Recuperamos los parametros del modelo de da�o para la matriz */
        alfam = PROPS(18)
        betam = PROPS(19)
        gammam = PROPS(20)
        ddamM=0.0
	  Pi=3.1416

C   Calcula los parametros de dmg: la variable wengyp, etam=1-damM(2), etap=damM(3), disip=damM(4)*/
	  if((wengy(2) >= damM(1))) then
			damM(1) = wengy(2)
		    etam = 1.0 
			damM(2) = 1.0 - etam 
			ddamM = 0.0
	  else
			wengyp = damM(1)
			bb = alfam+gammam*wengyp
			aa = dexp(-1.0*((wengyp-wengy(2))/bb)**2)
			var = (wengyp-wengy(2))/bb
			call func_erf(erfm,var,30)
	      write(*,*) 'kk'
	      etam = 1- (1/betam)*erfm
			damM(2) = 1.0 - etam
	      ddamM = -2.0*aa / (sqrt(Pi)*alfam*betam*bb)
			var = wengyp/bb
			call func_erf(erfp,var,30)
			damM(3) = 1- (1/betam)*erfp
			damM(4) = (bb/(sqrt(Pi)*betam))*(aa-1.0)+(1-etam)*wengyp
   	  end if
      return
	end subroutine DMGpseudo_matriz
C ------------------------------------------------------------------------------
C DMGvis_matriz
C Calcula el da�o viscoso en la matriz (Ju 1989)
C alfa y beta son parametros de la exponencial que regulan hdam
C mu es el parametro viscoso (velocidad de evolucion de la superfivie de da�o
C ------------------------------------------------------------------------------
 	subroutine dmgvis_matriz(PROPS,NPROPS,dtime,wengy,damM,hdam)
C
	dimension PROPS(NPROPS),damM(2),wengy(4)
	double precision PROPS, damM,dtime
	double precision wengy,hdam
	double precision double gtrial,incmum,etrial
      double precision double lminm,alfam,betam,mum

C Recuperamos los parametros del modelo de da�o viscoso para la matriz solida*/

        alfam = PROPS(18)
	  betam = PROPS(19)
	  mum = PROPS(20)
	  lminm = PROPS(21)

        etrial=sqrt(2*wengy(2))
	  hdam=0.0; gtrial=0.0

C   Calcula los parametros de dmg: la variable trial y si hay o no dmg */
C	damM(1)=rk

	  if((etrial>damM(1))) then
		incmum= mum*dtime
		gtrial=etrial-damM(1)
		damM(1)=(damM(1)+incmum*etrial)/(1+incmum) 
		if (etrial<lminm) then
			damM(2)=0.0
			hdam=0.0
		else
			hdam = alfam*betam*dexp(lminm-betam*etrial)
			damM(2) = damM(2)+incmum*gtrial*hdam
		end if
   	  end if
	  if (damM(2)>1.0) then 
		damM(2)=1.0
		hdam=0.0
	  end if

      return
	end subroutine dmgvis_matriz
C ------------------------------------------------------------------------------
C dmg_fibras
C Calcula el da�o en las fibras para el modelo CDM de Simo 1987c
C IMPORTANTE: Antes de utilizar la rutinas actualice la funcion da�o
C			que desee utilizar (var tipodam)
C Para unificar modelos de da�o, en damF[1-6] primera fibra, damF[7-12] segunda
C damF(1)=maxetrial_f1,damF(2)=dam_f1, damF(7)=maxetrial_f2,damF(8)=dam_f2
C ------------------------------------------------------------------------------
	subroutine dmg_fibras(PROPS,NPROPS,wengy,damM,DDAM1,DDAM2)
C
C Calcula los par�metros de da�o para la matriz segun el articulo
C
	dimension PROPS(NPROPS),damM(12),wengy(4)
	double precision PROPS, damM
	double precision WENGY,DDAM1,DDAM2
	integer load
	
	double precision double  bb,linm,lfim,betam,etrial1,etrial2
	integer tipodam

C Recuperamos los parametros del modelo de da�o para la matriz solida*/

	 linm = PROPS(21)
       lfim = PROPS(22)
       betam = PROPS(23)

C Definimos la funcion de da�o que queremos: 
C	tipodam=1 (Simo), tipodam=2 (cubica), tipodam=3 (sigmoide), tipodam=4 (sigmoide mod)
	 tipodam= PROPS(28) 

	 etrial1=sqrt(2.0 * wengy(3))
       ddam1=0.0
	 bb= (etrial1-linm)/(lfim-linm)

C   Calcula los parametros de dmg: la variable trial y si hay o no dmg para la familia 1*/

       if(( etrial1 > damM(1))) then
             damM(1)=etrial1
		if(( etrial1 > linm) .AND. (etrial1 < lfim)) then
			if (tipodam == 1) then
				bb= dexp(betam * (linm-lfim))
				aa= dexp(betam * (etrial1-lfim))
				damM(2) = 1.0-((1.0 - aa)/(1.0 -bb))
				ddam1 = betam * aa /bb
			else if (tipodam == 2) then
				damM(2) = bb**(2)*(1-betam*(bb**(2)-1))
				ddam1 = 4*betam*bb**(3) - 2*bb*(1+betam)
			else if (tipodam == 3) then
				expon=2.0*linm*(2.0*etrial1/lfim-1.0)
				ter1=exp(expon)
				ter2=etrial1*linm*exp(expon)-1.0
				ter3=1.0/(2+ter2)
				damM(2)=0.5*(1.0+ter2*ter3)
				ddam1=2.0*(1-damM(4))*(linm/lfim)*
	1				(4.0*etrial1*linm+lfim)*ter1*ter3
			else
				expon=2.0*betam*(2.0*bb-1.0)
				ter1=exp(expon)
				ter2=2.0*betam*bb*ter1+1.0
				ter3=2.0*betam*bb*ter1-1.0
				damM(4)=0.5*(1.0+ter2/ter3)
				ddam1=betam*ter1*(betam*bb*(betam*ter1*(bb*(8.0-
	1				16.0*betam*ter1)-8.0)-4.0)-2.0)/ter2**(2)
			end if
		else if (etrial1 > lfim) then 
		    damM(2) = 1.0  
			ddam1 = 0.0
          end if
	 end if
C Recuperamos los parametros del modelo de da�o para la segunda familia */
	 linm = PROPS(24)
       lfim = PROPS(25)
       betam = PROPS(26)

C Definimos la funcion de da�o que queremos: 
C	tipodam=1 (Simo), tipodam=2 (cubica), tipodam=3 (sigmoide), tipodam=4 (sigmoide mod)
	 tipodam= PROPS(29) 

C Calcula los parametros de dmg: la variable trial y si hay o no dmg para la familia 2*/

       etrial2=sqrt(2.0 * wengy(4))
       ddam2=0.0
	 bb= (etrial2-linm)/(lfim-linm)

	 if(( etrial2 > damM(7))) then
             damM(7)=etrial2
		if(( etrial2 > linm) .AND. (etrial2 < lfim)) then
			if (tipodam == 1) then
				bb= dexp(betam * (linm-lfim))
				aa= dexp(betam * (etrial2-lfim))
				damM(8) = 1.0-((1.0 - aa)/(1.0 -bb))
				ddam2 = betam * aa /bb 
			else if (tipodam == 2) then
				damM(8) = bb**(2)*(1-betam*(bb**(2)-1))
				ddam2 = 4*betam*bb**(3) - 2*bb*(1+betam)
			else if (tipodam == 3) then
				expon=2.0*linm*(2.0*etrial2/lfim-1.0)
				ter1=exp(expon)
				ter2=etrial2*linm*exp(expon)-1.0
				ter3=1.0/(2+ter2)
				damM(8)=0.5*(1.0+ter2*ter3)
				ddam2=2.0*(1-damM(4))*(linm/lfim)*
	1				(4.0*etrial2*linm+lfim)*ter1*ter3
			else
				expon=2.0*betam*(2.0*bb-1.0)
				ter1=exp(expon)
				ter2=2.0*betam*bb*ter1+1.0
				ter3=2.0*betam*bb*ter1-1.0
				damM(8)=0.5*(1.0+ter2/ter3)
				ddam2=betam*ter1*(betam*bb*(betam*ter1*(bb*(8.0-
	1				16.0*betam*ter1)-8.0)-4.0)-2.0)/ter2**(2)
			end if
		else if (etrial2 > lfim) then 
		    damM(8) = 1.0  
			ddam2 = 0.0
          end if
	 end if

	return
	end subroutine dmg_fibras
C
C ------------------------------------------------------------------------------
C dmg_continuom
C Calcula el da�o continuo y discontinuo en la fibras segun el articulo de Miehe 1997
C IMPORTANTE: Antes de utilizar la rutinas actualida la funcion da�o
C			que desee utilizar (var tipodam)
C Para unificar modelos de da�o, en damF[1-6] primera fibra, damF[7-12] segunda
C damF(1)=maxetrial_f1,damF(2)=dam_f1,damF(3)=maxetrial_f1_disco,damF(4)=dam_f1_disco,
C damF(5)=etrial_f1_cont,damF(6)=dam_f1_cont, 
C damF(7)=maxetrial_f2,damF(8)=dam_f2,damF(9)=maxetrial_f2_disco,damF(10)=dam_f2_disco,
C damF(11)=etrial_f2_cont,damF(12)=dam_f2_cont
C ------------------------------------------------------------------------------
 	subroutine DMGcont_fibras(PROPS,NPROPS,wengy,damM,ddam1,ddam2)
C
	dimension PROPS(NPROPS),damM(12),wengy(4)
	double precision PROPS, damM
	double precision wengy,ddam1,ddam2
	double precision ddama1,ddama2,ddamb1,ddamb2

	double precision etrial, dbinfinito, nubeta, beta, aa
	double precision linm, lfim, betam
	integer tipodam

C Definimos la funcion de da�o que queremos: 
C	tipodam=1 (Simo), tipodam=2 (cubica), tipodam=3 (sigmoide), tipodam=4 (sigmoide mod)
C Hay que definirlo en el fichero porque no hay espacio para constantes
	 tipodam=4

C Recuperamos los parametros del modelo de da�o para la primera familia */
	  linm = PROPS(23)
        lfim = PROPS(24)
        betam = PROPS(25)
        dbinfinito = PROPS(26)
	  nubeta = PROPS(27)
	  ddam1=0.0; ddama1=0.0; ddamb1=0.0; beta=0.0;
        etrial=sqrt(2*wengy(3))
	  bb= (etrial-linm)/(lfim-linm)
               
C Calculamos primero el damg discontinuo*/

	 if(( etrial > damM(3))) then  
	   damM(3)=etrial
		if((etrial > linm) .AND. (etrial < lfim)) then
			if (tipodam == 1) then
				bb= exp(betam * (linm-lfim))
				aa= dexp(betam * (etrial-lfim))
				damM(4) = 1.0 - ((1.0 - aa)/(1.0 -bb))
				ddama1 = betam * aa /bb 
			else if (tipodam == 2) then
				damM(4) = bb**(2)*(1-betam*(bb**(2)-1))
				ddama1 = 4*betam*bb**(3) - 2*bb*(1+betam)
			else if (tipodam == 3) then
				expon=2.0*linm*(2.0*etrial/lfim-1.0)
				ter1=exp(expon)
				ter2=etrial*linm*exp(expon)-1.0
				ter3=1.0/(2+ter2)
				damM(4)=0.5*(1.0+ter2*ter3)
				ddama1=2.0*(1-damM(4))*(linm/lfim)*
	1				(4.0*etrial*linm+lfim)*ter1*ter3
			else
				expon=2.0*betam*(2.0*bb-1.0)
				ter1=exp(expon)
				ter2=2.0*betam*bb*ter1+1.0
				ter3=2.0*betam*bb*ter1-1.0
				damM(4)=0.5*(1.0+ter2/ter3)
				ddama1=betam*ter1*(betam*bb*(betam*ter1*(bb*(8.0-
	1				16.0*betam*ter1)-8.0)-4.0)-2.0)/ter2**(2)
			end if
	    else if (etrial > lfim) then 
		    damM(4) = 1.0  
			ddama1 = 0.0
		end if
   	  end if
               
C Calculamos ahora el damg continuo*/

	 if(abs(PROPS(27)) <= 1.0E-6) then
	     damM(6) = 0.0
		 ddamb1 = 0.0
	 else
		 beta = damM(5) + abs(etrial - damM(1))      
		 damM(6) = dbinfinito*(1-dexp(-(beta/nubeta)))
		 if(etrial> damM(1)) then
			ddamb1 = dbinfinito*dexp(-(beta/nubeta))
		 else
			ddamb1 = -1.0*dbinfinito*dexp(-(beta/nubeta))
		 end if
		 damM(5)= beta
		 damM(1)= etrial
	 end if   
	 damM(2) =  damM(4) + damM(6)
	 ddam1 = ddama1 + ddamb1
	 if(damM(2) >= 1) then
		damM(2)= 1.0
		ddam1= 0.0
	 end if 

C Recuperamos los parametros del modelo de da�o para la segunda familia */
	  linm = PROPS(28)
        lfim = PROPS(29)
        betam = PROPS(30)
        dbinfinito = PROPS(31)
	  nubeta = PROPS(32)

	  ddam2=0.0; ddama2=0.0; ddamb2=0.0; beta=0.0;
        etrial=sqrt(2*wengy(4))
	  bb= (etrial-linm)/(lfim-linm)
               
C Calculamos primero el damg discontinuo*/

	  if(( etrial > damM(9))) then  
	    damM(9)=etrial
		if((etrial > linm) .AND. (etrial < lfim)) then
			if (tipodam == 1) then
				bb= exp(betam * (linm-lfim))
				aa= dexp(betam * (etrial-lfim))
				damM(10) = 1.0 - ((1.0 - aa)/(1.0 -bb))
				ddama2 = betam * aa /bb 
			else if (tipodam == 2) then
				damM(10) = bb**(2)*(1-betam*(bb**(2)-1))
				ddama2 = 4*betam*bb**(3) - 2*bb*(1+betam)
			else if (tipodam == 3) then
				expon=2.0*linm*(2.0*etrial/lfim-1.0)
				ter1=exp(expon)
				ter2=etrial*linm*exp(expon)-1.0
				ter3=1.0/(2+ter2)
				damM(10)=0.5*(1.0+ter2*ter3)
				ddama2=2.0*(1-damM(10))*(linm/lfim)*
	1				(4.0*etrial*linm+lfim)*ter1*ter3
			else
				expon=2.0*betam*(2.0*bb-1.0)
				ter1=exp(expon)
				ter2=2.0*betam*bb*ter1+1.0
				ter3=2.0*betam*bb*ter1-1.0
				damM(10)=0.5*(1.0+ter2/ter3)
				ddama2=betam*ter1*(betam*bb*(betam*ter1*(bb*(8.0-
	1				16.0*betam*ter1)-8.0)-4.0)-2.0)/ter2**(2)
			end if
	    else if (etrial > lfim) then 
		    damM(10) = 1.0  
			ddama2 = 0.0
		end if
   	  end if
               
C Calculamos ahora el damg continuo*/
	 if(abs(PROPS(32)) <= 1.0E-6) then
	     damM(12) = 0.0
		 ddamb1 = 0.0
	 else
		 beta = damM(11) + abs(etrial - damM(7))      
		 damM(12) = dbinfinito*(1-dexp(-(beta/nubeta)))
		 if(etrial> damM(7)) then
			ddamb2 = dbinfinito*dexp(-(beta/nubeta))
		 else
			ddamb2 = -1.0*dbinfinito*dexp(-(beta/nubeta))
		 end if
		 damM(11)= beta
		 damM(7)= etrial
   	  end if
	  damM(8) =  damM(10) + damM(12)
	  ddam2 = ddama2 + ddamb2
	 if(damM(8)>= 1) then
		damM(8)= 1.0
		ddam2= 0.0
	  end if        

      return
	end subroutine DMGcont_fibras
C
C ------------------------------------------------------------------------------
C DMGpseudo_fibras
C Calcula el da�o en las fibras para el modelo pseudoelastico de Ogden & Roxburg 1999 modificado por Bose
C IMPORTANTE: Para poder utilizar el esquema general del codigo para el da�o (tensiones y tensor elastico)
C			se envia al principal 1-eta y -damM
C ------------------------------------------------------------------------------
	subroutine DMGpseudo_fibras(PROPS,NPROPS,wengy,damM,ddam1,ddam2)
C Calcula los par�metros de da�o para la matriz segun el articulo
C
	dimension PROPS(NPROPS),damM(12),wengy(4)
	double precision PROPS, damM
	double precision wengy,ddam1,ddam2

      double precision alfam,betam,gammam,erfm,aa,erfp,bb
	double precision etaf1,etaf2,Pi,var

C Recuperamos los parametros del modelo de da�o para la primera familia de fibras */
        alfam = PROPS(21)
        betam = PROPS(22)
        gammam = PROPS(23)
        ddam1=0.0;ddam2=0.0
	  Pi=3.1416

C   Calcula los parametros de dmg: la variable wengyp, etam=1-damM(2), etap=damM(3), disip=damM(4)*/
	  if((wengy(3) >= damM(1))) then
			damM(1) = wengy(3)
		    etaf1 = 1.0  
			damM(2) = 1.0 - etaf1
			ddam1 = 0.0
	  else
			wengyp = damM(1)
			bb = alfam+gammam*wengyp
			aa = dexp(-1.0*((wengyp-wengy(3))/bb)**2)
			var = (wengyp-wengy(3))/bb
			call func_erf(erfm,var,30)
	        etaf1 = 1- (1/betam)*erfm
			damM(2) = 1.0 - etaf1
	        ddam1 = -2.0*aa / (sqrt(Pi)*betam*bb)
			var = wengyp/bb
			call func_erf(erfp,var,30)
			damM(3) = 1- (1/betam)*erfp
			damM(4) = (bb/(sqrt(Pi)*betam))*(aa-1.0)+(1-etaf1)*wengyp
   	  end if

C Recuperamos los parametros del modelo de da�o para la segunda familia */
	  alfam = PROPS(24)
        betam = PROPS(25)
        gammam = PROPS(26)

C   Calcula los parametros de dmg: la variable wengyp, etam=damM(2), etap=damM(3), disip=damM(4)*/
	  if((wengy(4) >= damM(7))) then
			damM(7) = wengy(4)
		    etaf2 = 1.0 
			damM(8) = 1.0 - etaf2
			ddam2 = 0.0
	  else
			wengyp = damM(7)
			bb = alfam+gammam*wengyp
			aa = dexp(-1.0*((wengyp-wengy(4))/bb)**2)
			var = (wengyp-wengy(4))/bb
			call func_erf(erfm,var,30)	        
			etaf2 = 1- (1/betam)*erfm
			damM(8) = 1.0 - etaf2
	        ddam2 = -2.0*aa / (sqrt(Pi)*betam*bb)
			var = wengyp/bb
			call func_erf(erfp,var,30)
			damM(9) = 1- (1/betam)*erfp
			damM(10) = (bb/(sqrt(Pi)*betam))*(aa-1.0)+(1-etaf2)*wengyp
   	  end if

	return
	end subroutine DMGpseudo_fibras
C ------------------------------------------------------------------------------
C DMGvis_fibras
C Calcula el da�o viscoso en la matriz (Ju 1989)
C alfa y beta son parametros de la exponencial que regulan hdam
C mu es el parametro viscoso (velocidad de evolucion de la superfivie de da�o
C ------------------------------------------------------------------------------
	subroutine DMGvis_fibras(PROPS,NPROPS,dtime,wengy,damM,hdam1,hdam2)
C
	dimension PROPS(NPROPS),damM(4),WENGY(4)
	double precision PROPS, damM, dtime
	double precision wengy,hdam1,hdam2
	double precision gtrial1,incmuf1,etrial1
      double precision lminf1,alfaf1,betaf1,muf1
	double precision gtrial2,incmuf2,etrial2
      double precision lminm2,lmaxm2,alfam2,betam2,mum2


C Recuperamos los parametros del modelo de da�o viscoso para la primera familia de fibras*/

        alfaf1 = PROPS(23)
	  betaf1 = PROPS(24)
	  muf1 = PROPS(25)
	  lminf1 = PROPS(26)

        etrial1=sqrt(2*wengy(3))
	  hdam1=0.0; gtrial1=0.0

C   Calcula los parametros de dmg: la variable trial y si hay o no dmg */
C	damM(1)=rk

	  if((etrial1>damM(1))) then
		incmuf1= muf1*dtime
		gtrial1=etrial1-damM(1)
		damM(1)=(damM(1)+incmuf1*etrial1)/(1+incmuf1) 
		if (etrial1<lminf1) then
			damM(2)=0.0
			hdam1=0.0
		else
			hdam1 = alfaf1*betaf1*dexp(lminf1-betaf1*etrial1)
			damM(2) = damM(2)+incmuf1*gtrial1*hdam1
		end if
   	  end if
	  if (damM(2)>1.0) then 
		damM(2)=1.0
		hdam1=0.0
	  end if
C
C Recuperamos los parametros del modelo de da�o viscoso para la segunda familia de fibras*/

        alfaf2 = PROPS(28)
	  betaf2 = PROPS(29)
	  muf2 = PROPS(30)
	  lminf2 = PROPS(31)

        etrial2=sqrt(2*wengy(4))
	  hdam2=0.0; gtrial2=0.0

C   Calcula los parametros de dmg: la variable trial y si hay o no dmg */
C	damM(1)=rk

	  if((etrial2>damM(7))) then
		incmuf2= muf2*dtime
		gtrial2=etrial2-damM(7)
		damM(7)=(damM(7)+incmuf2*etrial2)/(1+incmuf2) 
		if (etrial2<lminf2) then
			damM(8)=0.0
			hdam2=0.0
		else
			hdam2 = alfaf2*betaf2*dexp(lminf2-betaf2*etrial2)
			damM(8) = damM(8)+incmuf2*gtrial2*hdam2
		end if
   	  end if
	  if (damM(8)>1.0) then 
		damM(8)=1.0
		hdam2=0.0
	  end if

      return
	end subroutine DMGvis_fibras
C ------------------------------------------------------------------------------
C kelvinVoigt
C Actualiza sigma y dmat siguiendo un comportamineto viscoelastico
C tipo Kelvin-Voigt con diferente comportamineto matriz y cada fibra
C ------------------------------------------------------------------------------
	SUBROUTINE kelvinVoigt(PROPS,NPROPS,dtime,Hn,Sn,jac,poct,Se,Si,
	1   Sa1,Sa2,Sv,inva,ntens,FG,Ctot,Cv,Ci,Ca1,Ca2)
C
	real*8 PROPS(NPROPS),Hn(NTENS,3),Sn(NTENS,3),Sv(NTENS,1)
	real*8 Se(NTENS,1),Si(NTENS,1),Sa1(NTENS,1),Sa2(NTENS,1)
	real*8 FG(3,3),Ca2(NTENS,NTENS),Ci(NTENS,NTENS),Ca1(NTENS,NTENS) 
      real*8 Ctot(NTENS,NTENS),Cv(NTENS,NTENS),jac,jacg,dtime,inva(4)
	real*8 Taoi(NTENS,1),Taoa1(NTENS,1),Taoa2(NTENS,1),Tao(NTENS,1)
	real*8 Seqi(NTENS,1),Seqa1(NTENS,1),Seqa2(NTENS,1),Stot(NTENS,3)
	real*8 invFG(3,3),expon(3),gamma(3),Ht(NTENS,3),Hngh(NTENS,1)
	real*8 Hngg,trqvisco,qvisco(NTENS,1),ttau(3)
	real*8 tproyec(NTENS,NTENS),uni(NTENS,1)
	integer i

C	
	 Hngg=1.0;jacg=1.0;Hngh=0.0;Ht=0.0
	 uni=0.0
	 Taoi=jac*Si; Taoa1=jac*Sa1; Taoa2=jac*Sa2;
	 call inv(FG,invFG)

C	 Calculamos el tiron de tau (ojo, FG ya es desviador) para obtner Sequilibrio

	 call pull(NTENS,invFG,Taoi,Seqi)	 
	 call pull(NTENS,invFG,Taoa1,Seqa1)
	 call pull(NTENS,invFG,Taoa2,Seqa2)

C	 Calculamos Sequilibrio^gorro a partir de Sequilibri^barra

	 Stot(:,1)=Seqi(:,1); Stot(:,2)=Seqa1(:,1); Stot(:,3)=Seqa2(:,1);
	 jacg=jac**(-2.0/3.0)
	 Stot= Stot/jacg

C	 Adaptamos las variables histerm para el instante actual

	 if(abs(PROPS(33)-1.0).LT.1.0E-6) then
		gamma(1)=PROPS(34); gamma(2)=PROPS(36); gamma(3)=PROPS(38);
		ttau(1)=PROPS(35); ttau(2)=PROPS(37); ttau(3)=PROPS(39);
	 elseif(abs(PROPS(33)-3.0).LT.1.0E-6) then
		 gamma(1)= PROPS(34)*exp(-1.0*(inva(1)-3.0)*PROPS(35))
		 gamma(2)= PROPS(38)*exp(-1.0*(inva(3)-1.0)*PROPS(39))
		 gamma(3)= PROPS(42)*exp(-1.0*(inva(4)-1.0)*PROPS(43))
		 ttau(1)= PROPS(36)*exp((inva(1)-3.0)*PROPS(37))
		 ttau(2)= PROPS(40)*exp((inva(3)-1.0)*PROPS(41))
		 ttau(3)= PROPS(44)*exp((inva(4)-1.0)*PROPS(45))
       end if

	 do i=1,3
         expon(i)=exp(-dtime/(2*ttau(i)))
	   Ht(:,i)=expon(i)*(expon(i)*Hn(:,i)-Sn(:,i))
	   Hn(:,i)=Ht(:,i)+expon(i)*Stot(:,i)
	   Hngh(:,1)=Hngh(:,1)+gamma(i)*Ht(:,i)
	   Hngg= Hngg-gamma(i)*(1-expon(i))
	 enddo

C	 Actualizamos Sn
	 
	 Sn=Stot

C	 Calculamos el empuje del termino visco Hngh
	 
	 call pull(NTENS,FG,Hngh,qvisco)
	 qvisco= qvisco*jacg
	 trqvisco=sum((/(qvisco(i,1),i=1,3)/))/3.0
	 do i=1,3
         qvisco(i,1)= qvisco(i,1) - trqvisco
	 enddo
	 tao=(1.0-gamma(1)+gamma(1)*(expon(1)))*Taoi+
	1     (1.0-gamma(2)+gamma(2)*(expon(2)))*Taoa1+ 
     2     (1.0-gamma(3)+gamma(3)*(expon(3)))*Taoa2 
       Se=Sv+(1.0/jac)*(tao+qvisco)

C	 Calculamos el tensor de comportamiento

	 Ctot=(1.0-gamma(1)+gamma(1)*(expon(1)))*Ci+
	1     (1.0-gamma(2)+gamma(2)*(expon(2)))*Ca1+ 
     2     (1.0-gamma(3)+gamma(3)*(expon(3)))*Ca2 

	 uni(1:3,1)=1.0
	 call ten_proy(tproyec,NTENS)

	 Ctot=Cv+Ctot+(2.0/(3.0*jac))*(trqvisco*tproyec -
	1       (matmul(qvisco,transpose(uni))+
     2       matmul(uni,transpose(qvisco))))

	return
	end subroutine kelvinVoigt
C ------------------------------------------------------------------------------
C Maxwell
C Actualiza sigma y dmat siguiendo un comportamineto viscoelastico
C tipo Maxwell con diferente comportamineto matriz y cada fibra
C ------------------------------------------------------------------------------
	SUBROUTINE Maxwell(PROPS,NPROPS,dtime,Hn,Sn,jac,poct,Se,Si,
	1   Sa1,Sa2,Sv,ntens,FG,Ctot,Cv,Ci,Ca1,Ca2)
C
	real*8 PROPS(NPROPS),Hn(NTENS,3),Sn(NTENS,3),Sv(NTENS,1)
	real*8 Se(NTENS,1),Si(NTENS,1),Sa1(NTENS,1),Sa2(NTENS,1)
	real*8 FG(3,3),Ca2(NTENS,NTENS),Ci(NTENS,NTENS),Ca1(NTENS,NTENS) 
      real*8 Ctot(NTENS,NTENS),Cv(NTENS,NTENS)


	return
	end subroutine Maxwell 

C ------------------------------------------------------------------------------
C genera_fibras_vena
C Genera la direcci�n de las fibras para un cilindro con una orientacion dada
C a partir de un eje director y un pinto de referencia
C ------------------------------------------------------------------------------
	SUBROUTINE genera_fibras_vena(COORDS,PROPS,NPROPS,na0,nb0)

C	Calcula los par�metros de da�o para la matriz segun el articulo

	real*8 COORDS(3),PROPS(NPROPS)
	real*8 na0(3,1), nb0(3,1)
	real*8 phi, dir(3), o(3), op(3), dop(3)
	real*8 modul
	
       phi = (PROPS(56)*3.1416)/180.0
C      phi = (90.0*3.1416)/180.0
	 dir(1) = PROPS(50)
	 dir(2) = PROPS(51)
	 dir(3) = PROPS(52)

	 o(1) = PROPS(53)
	 o(2) = PROPS(54)
	 o(3) = PROPS(55)

	 op = o - COORDS 
       
	 call prod_vect(dir,op,dop)
	 modul = sqrt(dot_product(dop,dop))
	 dop = dop/modul

	 na0(1,1) = dir(1) * cos(phi) + dop(1) * sin(phi)
	 na0(2,1) = dir(2) * cos(phi) + dop(2) * sin(phi)
	 na0(3,1) = dir(3) * cos(phi) + dop(3) * sin(phi)

	 nb0(1,1) = dir(1) * cos(phi) - dop(1) * sin(phi)
	 nb0(2,1) = dir(2) * cos(phi) - dop(2) * sin(phi)
	 nb0(3,1) = dir(3) * cos(phi) - dop(3) * sin(phi)

	 modul = sqrt(dot_product(na0(:,1),na0(:,1)))
	 na0 = na0/modul
	 modul = sqrt(dot_product(nb0(:,1),nb0(:,1)))
	 nb0 = nb0/modul

	return
	end subroutine genera_fibras_vena

C ------------------------------------------------------------------------------
C prod_vect
C Calcula el producto vectorial de dos vectores de orden 3
C ------------------------------------------------------------------------------

	subroutine prod_vect(u,v,w)
	real*8 u(3),v(3),w(3)

	w(1) = u(2) * v(3) - u(3) * v(2)
	w(2) = u(3) * v(1) - u(1) * v(3)
	w(3) = u(1) * v(2) - u(2) * v(1)

	return
	end subroutine prod_vect
C ------------------------------------------------------------------------------
C empuje
C Calcula el empuje de un tensor de orden 2 en coordenadas
C contravariantes
C ------------------------------------------------------------------------------

	subroutine pull(NTENS,dfgrd,tensor1,tensor2)
	real*8 dfgrd(3,3),tensor1(NTENS,1),tensor2(NTENS,1)
	real*8 tensora1(NTENS/2,NTENS/2),tensora2(NTENS/2,NTENS/2)
	real*8 ten(NTENS/2,NTENS/2)

	tensora1(1,1)=tensor1(1,1); 
	tensora1(2,2)=tensor1(2,1);
	tensora1(3,3)=tensor1(3,1); 
	tensora1(1,2)=tensor1(4,1);
	tensora1(2,1)=tensor1(4,1);
	tensora1(1,3)=tensor1(5,1);
	tensora1(3,1)=tensor1(5,1);
	tensora1(2,3)=tensor1(6,1);
	tensora1(3,2)=tensor1(6,1);

	ten=matmul(dfgrd,tensora1)
	tensora2=matmul(ten,transpose(dfgrd))

	tensor2(1,1)=tensora2(1,1); tensor2(2,1)=tensora2(2,2); 
	tensor2(3,1)=tensora2(3,3); tensor2(4,1)=tensora2(1,2);
	tensor2(5,1)=tensora2(1,3); tensor2(6,1)=tensora2(2,3);  

	return
	end subroutine pull

C ------------------------------------------------------------------------------
C ten_proy
C Calcula el tensor de proyeccion de orden 4: Id-(1/3) 1*1
C ------------------------------------------------------------------------------

	subroutine ten_proy(tproyec,NTENS)
	real*8 tproyec(NTENS,NTENS)
	real*8 uno(NTENS,NTENS),Id(NTENS,NTENS)

	 uno=0.0
       uno(1:3,1)=1.0
	 tproyec=0.0
       call eye(NTENS,Id)
       Id(4,4)=0.5
       Id(5,5)=0.5
       Id(6,6)=0.5
	 tproyec=Id-(1.0/3.0)*uno

	return
	end subroutine ten_proy
C ------------------------------------------------------------------------------
C erf
C Calcula la funcion error para un numero n de terminos de la serie
C ------------------------------------------------------------------------------

	subroutine func_erf(erfi,var,n)
	double precision erfi,var
	integer n

	double precision aux,Pi,prod
	integer i,j
	 Pi=3.1416
	 aux=0.0 
	 do i=3,n
		call fact(i-1,prod)
		a=0.5*(2*i-1)*prod
		j=(2*i-1)
	 	aux = aux+var**j/a
	 enddo
	 erfi = 1.0/sqrt(Pi)*(2.0*var+2.0/3.0*var**3+aux)
	return
	end subroutine func_erf
C ------------------------------------------------------------------------------
C factorial
C Calcula el factorial n
C ------------------------------------------------------------------------------
	subroutine fact(n,aux)
	double precision aux
	integer n,i

       aux = 1.0
       do i = 2, n
             aux = aux * i
       enddo
	return
	end subroutine fact
