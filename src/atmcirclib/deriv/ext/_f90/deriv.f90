!     ****************************************************************
!     * SUBROUTINE SECTION: ELEMENTARY VECTOR ANALYTICAL EXPRESSIONS *
!     ****************************************************************

! SR module deriv

! SR implicit none

! SR contains
!     -----------------------------------------------------------------
!     Gradient
!     -----------------------------------------------------------------

      subroutine grad(gradx,grady,scalar, &
                     vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

!     Calculate the two vector components <gradx> and <grady> of the
!     scalar field <scalar>.  The vertical coordinate is specified in
!     <vert>, the grid in <xmin,ymin,dx,dy,nx,ny,nz,mdv>.

      implicit none

!     Declaration of subroutine parameters
      integer nx,ny,nz
!f2py intent(in) nx,ny,nz
      real    gradx(nx,ny,nz)
!f2py intent(out) gradx
      real    grady(nx,ny,nz)
!f2py intent(out) grady
      real    scalar(nx,ny,nz)
!f2py intent(in)  scalar
      real    vert(nx,ny,nz)
!f2py intent(in) vert
      real    xmin,ymin,dx,dy
!f2py intent(in) xmin,ymin,dx,dy
      real    mdv
!f2py intent(in) mdv

!     Calculate the derivatives in x and y direction
      call deriv (gradx,scalar,'x',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
      call deriv (grady,scalar,'y',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

      end subroutine grad

!     -----------------------------------------------------------------
!     Advection of scalar field (horizontal)
!     -----------------------------------------------------------------

      subroutine hadv(adv3,u3,v3,f3,vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

!     Calculate the horizontal advection <div> of the scalar field f3 by <uu>
!     <vv>. The vertical coordinate is specified in <vert>, the grid
!     in <xmin,ymin,dx,dy,nx,ny,nz,mdv>.

      implicit none

!     Declaration of subroutine parameters
      integer    nx,ny,nz
!f2py intent(in) nx,ny,nz
      real       u3(nx,ny,nz)
!f2py intent(in)  u3
      real       v3(nx,ny,nz)
!f2py intent(in)  v3
      real       f3(nx,ny,nz)
!f2py intent(in) f3

      real       adv3(nx,ny,nz)
!f2py intent(out) adv3
      real       vert(nx,ny,nz)
!f2py intent(in) vert
      real       xmin,ymin,dx,dy
!f2py intent(in) xmin,ymin,dx,dy
      real       mdv
!f2py intent(in) mdv

!     Mathematical and physical parameters
      real       pi180
      parameter  (pi180=3.141592654/180.)
      real       zerodiv
      parameter  (zerodiv=0.00000001)
      real       eps
      parameter  (eps=0.01)

!     Auxiliary variables
      integer    i,j,k
      real       dfdy(nx,ny,nz)
      real       dfdx(nx,ny,nz)

!     Calculate the derivatives in x and y direction
      call deriv (dfdy, f3,'y',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
      call deriv (dfdx, f3,'x',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

!     Calculate the advection by u and v
      do i=1,nx
       do j=1,ny
        do k=1,nz
         if( (abs(u3(i,j,k)-mdv).gt.eps)  .and. &
             (abs(v3(i,j,k)-mdv).gt.eps) ) then

          adv3(i,j,k) = u3(i,j,k)*dfdx(i,j,k) + v3(i,j,k)*dfdy(i,j,k)

         endif
        enddo
       enddo
      enddo

     end subroutine hadv

!     -----------------------------------------------------------------
!     Rotation
!     -----------------------------------------------------------------

      subroutine rot(vort3,u3,v3, &
                    vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

!     Calculate the vertical component <vort> of the vector field <uu>
!     <vv>. The vertical coordinate is specified in <vert>, the grid
!     in <xmin,ymin,dx,dy,nx,ny,nz,mdv>.

      implicit none

!     Declaration of subroutine parameters
      integer    nx,ny,nz
!f2py intent(in) nx,ny,nz
      real       u3(nx,ny,nz)
!f2py intent(in) u3
      real       v3(nx,ny,nz)
!f2py intent(in) v3
      real       vort3(nx,ny,nz)
!f2py intent(out) vort3
      real       vert(nx,ny,nz)
!f2py intent(in) vert
      real       xmin,ymin,dx,dy
!f2py intent(in) xmin,ymin,dx,dy
      real       mdv
!f2py intent(in) mdv

!     Mathematical and physical parameters
      real       pi180
      parameter  (pi180=3.141592654/180.)
      real       zerodiv
      parameter  (zerodiv=0.00000001)
      real       eps
      parameter  (eps=0.01)

!     Auxiliary variables
      integer    i,j,k
      real       ucos(nx,ny,nz)
      real       coslat(nx,ny)
      real       dvdx(nx,ny,nz)
      real       dudy(nx,ny,nz)
      real       lat

!     Calculate Cos factor: correction for spherical coordinates
      do i=1,nx
         do j=1,ny
            lat=ymin+real(j-1)*dy
            coslat(i,j)=cos(pi180*lat)
         enddo
      enddo

!     Derivatives of zonal velocity U: correction for spherical coordinates
      do i=1,nx
         do j=1,ny
            do k=1,nz
               if(abs(u3(i,j,k)-mdv).gt.eps) then
                  ucos(i,j,k)=u3(i,j,k)*coslat(i,j)
               endif
            enddo
         enddo
      enddo
      call deriv (dudy, ucos,'y',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
      do i=1,nx
         do j=1,ny
            do k=1,nz
               if (abs(dudy(i,j,k)-mdv).gt.eps) then
                  dudy(i,j,k)=dudy(i,j,k)/(coslat(i,j)+zerodiv)
               endif
            enddo
         enddo
      enddo

!     Derivatives of meridional velocity V
      call deriv (dvdx, v3,'x',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

!     Calculate rotation of vector field
      do i=1,nx
         do j=1,ny
            do k=1,nz
               if ((abs(dvdx(i,j,k)-mdv).gt.eps).and. &
                  (abs(dudy(i,j,k)-mdv).gt.eps)) then
                  vort3(i,j,k)=dvdx(i,j,k)-dudy(i,j,k)
               else
                  vort3(i,j,k)=mdv
               endif
            enddo
         enddo
      enddo

      end subroutine rot

!     -----------------------------------------------------------------
!     Divergence
!     -----------------------------------------------------------------

      subroutine div(div3,u3,v3, &
                    vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

!     Calculate the divergence <div> of the horizontal vector field <uu>
!     <vv>. The vertical coordinate is specified in <vert>, the grid
!     in <xmin,ymin,dx,dy,nx,ny,nz,mdv>.

      implicit none

!     Declaration of subroutine parameters
      integer    nx,ny,nz
!f2py intent(in) nx,ny,nz
      real       u3(nx,ny,nz)
!f2py intent(in) u3
      real       v3(nx,ny,nz)
!f2py intent(in) v3
      real       div3(nx,ny,nz)
!f2py intent(out) div3
      real       vert(nx,ny,nz)
!f2py intent(in) vert
      real       xmin,ymin,dx,dy
!f2py intent(in) xmin,ymin,dx,dy
      real       mdv
!f2py intent(in) mdv

!     Mathematical and physical parameters
      real       pi180
      parameter  (pi180=3.141592654/180.)
      real       zerodiv
      parameter  (zerodiv=0.00000001)
      real       eps
      parameter  (eps=0.01)

!     Auxiliary variables
      integer    i,j,k
      real       vcos(nx,ny,nz)
      real       coslat(nx,ny)
      real       dvdy(nx,ny,nz)
      real       dudx(nx,ny,nz)
      real       lat

!     Calculate Cos factor: correction for spherical coordinates
      do i=1,nx
         do j=1,ny
            lat=ymin+real(j-1)*dy
            coslat(i,j)=cos(pi180*lat)
         enddo
      enddo

!     Derivatives of meridional velocity V: correction for spherical coordinates
      do i=1,nx
         do j=1,ny
            do k=1,nz
               if(abs(v3(i,j,k)-mdv).gt.eps) then
                  vcos(i,j,k)=v3(i,j,k)*coslat(i,j)
               endif
            enddo
         enddo
      enddo
      call deriv (dvdy, vcos,'y',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
      do i=1,nx
         do j=1,ny
            do k=1,nz
               if (abs(dvdy(i,j,k)-mdv).gt.eps) then
                  dvdy(i,j,k)=dvdy(i,j,k)/(coslat(i,j)+zerodiv)
               endif
            enddo
         enddo
      enddo

!     Derivatives of zonal velocity U
      call deriv (dudx, u3,'x',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

!     Calculate divergence of vector field
      do i=1,nx
         do j=1,ny
            do k=1,nz
               if ((abs(dvdy(i,j,k)-mdv).gt.eps).and. &
                (abs(dudx(i,j,k)-mdv).gt.eps)) then
                  div3(i,j,k)=dvdy(i,j,k)+dudx(i,j,k)
               else
                  div3(i,j,k)=mdv
               endif
            enddo
         enddo
      enddo

      end subroutine div

!     -----------------------------------------------------------------
!     Vector gradient
!!     -----------------------------------------------------------------

      subroutine gradvec(a,b,c,scalar,dir,comp, &
                     vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

!     Calculate the component "comp" of the gradient of
!     vector (a,b,c) in direction "dir".
!     The vertical coordinate is specified in
!     <vert>, the grid in <xmin,ymin,dx,dy,nx,ny,nz,mdv>.

      implicit none

!     Declaration of subroutine parameters
      integer nx,ny,nz
!f2py intent(in) nx,ny,nz
      integer i,j,k
      real    a(nx,ny,nz)
!f2py intent(in) a
      real    b(nx,ny,nz)
!f2py intent(in) b
      real    c(nx,ny,nz)
!f2py intent(in) c
      real    dd(nx,ny,nz)	! derivative (only one is needed)
      real    scalar(nx,ny,nz)
!f2py intent(out) scalar
      real    tanlat(nx,ny)
      real    vert(nx,ny,nz)
!f2py intent(in) vert
      real    xmin,ymin,dx,dy
!f2py intent(in) xmin,ymin,dx,dy
      real    mdv
!f2py intent(in) mdv
      real    lat

      character  dir
!f2py intent(in) dir
      character  comp
!f2py intent(in) comp

!     Mathematical and physical parameters
      real       pi180
      parameter  (pi180=3.141592654/180.)
      real       zerodiv
      parameter  (zerodiv=0.00000001)
      real       eps
      parameter  (eps=0.01)
      real       rearth
      parameter  (rearth=6.3675*10**6)	! m

!     derivative in x-direction
      if (dir .eq. 'x') then

	   !  Calculate tan factor
      	   do i=1,nx
             do j=1,ny
               lat=ymin+real(j-1)*dy
               tanlat(i,j)=tan(pi180*lat)
             enddo
           enddo

	   ! x-component
           if (comp .eq. 'x') then
              call deriv (dd, a,'x',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
              do k=1,nz
         	do j=1,ny
            	  do i=1,nx
                    if ((abs(dd(i,j,k)-mdv).gt.eps).and. &
                        (abs(b(i,j,k)-mdv).gt.eps).and. &
			(abs(c(i,j,k)-mdv).gt.eps)) then
			scalar(i,j,k) = dd(i,j,k) - tanlat(i,j) * b(i,j,k) / rearth + c(i,j,k) / rearth
                    else
                        scalar(i,j,k)=mdv
                    endif
                  enddo
                enddo
              enddo

	   ! y-component
	   else if (comp .eq. 'y') then
              call deriv (dd, b,'x',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
              do k=1,nz
         	do j=1,ny
            	  do i=1,nx
                    if ((abs(dd(i,j,k)-mdv).gt.eps).and. &
                        (abs(a(i,j,k)-mdv).gt.eps)) then
			scalar(i,j,k) = dd(i,j,k) + tanlat(i,j) * a(i,j,k) / rearth
                    else
                        scalar(i,j,k)=mdv
                    endif
                  enddo
                enddo
              enddo

           ! z-component
	   else if ((comp .eq. 'z').or. &
                    (comp .eq. 'th').or.&
                    (comp .eq. 'p').or.&
                    (comp .eq. 'm')) then

              call deriv (dd, c,'x',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
              do k=1,nz
         	do j=1,ny
            	  do i=1,nx
                    if ((abs(dd(i,j,k)-mdv).gt.eps).and. &
                        (abs(a(i,j,k)-mdv).gt.eps)) then
			scalar(i,j,k) = dd(i,j,k) - a(i,j,k) / rearth
                    else
                        scalar(i,j,k)=mdv
                    endif
                  enddo
                enddo
              enddo

           ! invalid component
           else
	 	  print*,'Invalid component... Stop'
           	  stop
	   end if

!     derivative in y-direction
      else if (dir .eq. 'y') then

	   ! x-component
           if (comp .eq. 'x') then
              call deriv (dd, a,'y',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
              do k=1,nz
         	do j=1,ny
            	  do i=1,nx
                    if (abs(dd(i,j,k)-mdv).gt.eps) then
			scalar(i,j,k) = dd(i,j,k)
                    else
                        scalar(i,j,k)=mdv
                    endif
                  enddo
                enddo
              enddo

	   ! y-component
	   else if (comp .eq. 'y') then
              call deriv (dd, b,'y',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
              do k=1,nz
         	do j=1,ny
            	  do i=1,nx
                    if ((abs(dd(i,j,k)-mdv).gt.eps).and. &
			(abs(c(i,j,k)-mdv).gt.eps)) then
			scalar(i,j,k) = dd(i,j,k) + c(i,j,k) / rearth
                    else
                        scalar(i,j,k)=mdv
                    endif
                  enddo
                enddo
              enddo

           ! z-component
	   else if ((comp .eq. 'z').or. &
                    (comp .eq. 'th').or.&
                    (comp .eq. 'p').or.&
                    (comp .eq. 'm')) then

              call deriv (dd, c,'y',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
              do k=1,nz
         	do j=1,ny
            	  do i=1,nx
                    if ((abs(dd(i,j,k)-mdv).gt.eps).and. &
			(abs(b(i,j,k)-mdv).gt.eps)) then
			scalar(i,j,k) = dd(i,j,k) - b(i,j,k) / rearth
                    else
                        scalar(i,j,k)=mdv
                    endif
                  enddo
                enddo
              enddo

           ! invalid component
           else
	 	  print*,'Invalid component... Stop'
           	  stop
	   end if

!     derivative in vertical direction
      else if ((dir.eq.'z').or. &
               (dir.eq.'th').or. &
               (dir.eq.'p').or. &
               (dir.eq.'m')) then

	   ! x-component
           if (comp .eq. 'x') then

              call deriv (dd,a,'z',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
              do k=1,nz
         	do j=1,ny
            	  do i=1,nx
                    if (abs(dd(i,j,k)-mdv).gt.eps) then
			scalar(i,j,k) = dd(i,j,k)
                    else
                        scalar(i,j,k)=mdv
                    endif
                  enddo
                enddo
              enddo

	   ! y-component
	   else if (comp .eq. 'y') then

              call deriv (dd,b,'z',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
              do k=1,nz
         	do j=1,ny
            	  do i=1,nx
                    if (abs(dd(i,j,k)-mdv).gt.eps) then
			scalar(i,j,k) = dd(i,j,k)
                    else
                        scalar(i,j,k)=mdv
                    endif
                  enddo
                enddo
              enddo

           ! z-component
	   else if ((comp .eq. 'z').or. &
                    (comp .eq. 'th').or.&
                    (comp .eq. 'p').or.&
                    (comp .eq. 'm')) then

              call deriv (dd,c,'z',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
              do k=1,nz
         	do j=1,ny
            	  do i=1,nx
                    if (abs(dd(i,j,k)-mdv).gt.eps) then
			scalar(i,j,k) = dd(i,j,k)
                    else
                        scalar(i,j,k)=mdv
                    endif
                  enddo
                enddo
              enddo

           ! invalid component
           else
	 	  print*,'Invalid component... Stop'
           	  stop
	   end if

!     invalid direction
      else
	   print*,'Invalid direction of derivative... Stop'
           stop
      end if


      end subroutine gradvec


!     -----------------------------------------------------------------
!     Material derivative ([A NABLA] B)
!     -----------------------------------------------------------------

      subroutine gradmat(A1,A2,A3,B1,B2,B3,scalar,comp, &
                     vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

!     Calculate the component "comp" of the material derivative
!     of B along A ([A NABLA] B).
!     The vertical coordinate is specified in
!     <vert>, the grid in <xmin,ymin,dx,dy,nx,ny,nz,mdv>.

      implicit none

!     Declaration of subroutine parameters
      integer nx,ny,nz
!f2py intent(in) nx,ny,nz
      integer i,j,k
      real    A1(nx,ny,nz)
!f2py intent(in) A1
      real    A2(nx,ny,nz)
!f2py intent(in) A2
      real    A3(nx,ny,nz)
!f2py intent(in) A3
      real    B1(nx,ny,nz)
!f2py intent(in) B1
      real    B2(nx,ny,nz)
!f2py intent(in) B2
      real    B3(nx,ny,nz)
!f2py intent(in) B3
      real    ddx(nx,ny,nz)
      real    ddy(nx,ny,nz)
      real    ddz(nx,ny,nz)
      real    scalar(nx,ny,nz)
!f2py intent(out) scalar
      real    vert(nx,ny,nz)
!f2py intent(in) vert
      real    xmin,ymin,dx,dy
!f2py intent(in) xmin,ymin,dx,dy
      real    mdv
!f2py intent(in) mdv
      real    lat

      character  comp
!f2py intent(in) comp

!     Mathematical and physical parameters
      real       pi180
      parameter  (pi180=3.141592654/180.)
      real       zerodiv
      parameter  (zerodiv=0.00000001)
      real       eps
      parameter  (eps=0.01)
      real       rearth
      parameter  (rearth=6.3675*10**6)	! m

!     x component
      if (comp .eq. 'x') then
        ! i dB/dx
      	call gradvec(B1,B2,B3,ddx,'x','x', &
                     vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
        ! i dB/dy
      	call gradvec(B1,B2,B3,ddy,'y','x', &
                     vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
        ! i dB/dz
      	call gradvec(B1,B2,B3,ddz,'z','x', &
                     vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

!     y component
      else if (comp .eq. 'y') then

        ! j dB/dx
      	call gradvec(B1,B2,B3,ddx,'x','y', &
                     vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
        ! j dB/dy
      	call gradvec(B1,B2,B3,ddy,'y','y', &
                     vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
        ! j dB/dz
      	call gradvec(B1,B2,B3,ddz,'z','y', &
                     vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

!     z component
      else if ((comp.eq.'z').or. &
               (comp.eq.'th').or. &
               (comp.eq.'p').or. &
               (comp.eq.'m')) then

        ! k dB/dx
      	call gradvec(B1,B2,B3,ddx,'x','z', &
                     vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
        ! k dB/dy
      	call gradvec(B1,B2,B3,ddy,'y','z', &
                     vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
        ! k dB/dz
      	call gradvec(B1,B2,B3,ddz,'z','z', &
                     vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

!     invalid direction
      else
	   print*,'Invalid component... Stop'
           stop
      end if

      do k=1,nz
        do j=1,ny
          do i=1,nx
            if ((abs(ddx(i,j,k)-mdv).gt.eps).and. &
                (abs(ddy(i,j,k)-mdv).gt.eps).and. &
	        (abs(ddz(i,j,k)-mdv).gt.eps).and. &
		(abs(A1(i,j,k)-mdv).gt.eps).and. &
                (abs(A2(i,j,k)-mdv).gt.eps).and. &
	        (abs(A3(i,j,k)-mdv).gt.eps)) then
			scalar(i,j,k) = A1(i,j,k)*ddx(i,j,k) + A2(i,j,k)*ddy(i,j,k) + &
                                        A3(i,j,k)*ddz(i,j,k)
             else
                     scalar(i,j,k)=mdv
             endif
           enddo
         enddo
       enddo


      end subroutine gradmat

!     -----------------------------------------------------------------
!     Deformation F1 and F2
!     -----------------------------------------------------------------

      subroutine deformation(u,v,F1,F2, &
                     vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

!     Calculate the two deformation fields F1 and F2
!     from the horizontal velocities u and v.
!     The vertical coordinate is specified in
!     <vert>, the grid in <xmin,ymin,dx,dy,nx,ny,nz,mdv>.

      implicit none

!     Declaration of subroutine parameters
      integer nx,ny,nz
!f2py intent(in) nx,ny,nz
      real    u(nx,ny,nz)
!f2py intent(in) u
      real    v(nx,ny,nz)
!f2py intent(in) v
      real    dudx(nx,ny,nz)
      real    dudy(nx,ny,nz)
      real    dvdx(nx,ny,nz)
      real    dvdy(nx,ny,nz)
      real    F1(nx,ny,nz)
!f2py intent(out) F1
      real    F2(nx,ny,nz)
!f2py intent(out) F2
      real    coslat(nx,ny)
      real    ucos(nx,ny,nz)
      real    vcos(nx,ny,nz)
      real    vert(nx,ny,nz)
!f2py intent(in) vert
      real    xmin,ymin,dx,dy
!f2py intent(in) xmin,ymin,dx,dy
      real    mdv
!f2py intent(in) mdv
      real    lat
      integer i,j,k

!     Mathematical and physical parameters
      real       pi180
      parameter  (pi180=3.141592654/180.)
      real       zerodiv
      parameter  (zerodiv=0.00000001)
      real       eps
      parameter  (eps=0.01)

!     Calculate cos factor
      do i=1,nx
         do j=1,ny
            lat=ymin+real(j-1)*dy
            coslat(i,j)=cos(pi180*lat)
         enddo
      enddo

!     Calculate cos weighted velocities
      do k=1,nz
        do j=1,ny
          do i=1,nx
               if(abs(v(i,j,k)-mdv).gt.eps) then
                  vcos(i,j,k)=v(i,j,k)*coslat(i,j)
               endif
               if(abs(u(i,j,k)-mdv).gt.eps) then
                  ucos(i,j,k)=u(i,j,k)*coslat(i,j)
               endif
            enddo
         enddo
      enddo

!     Calculate y-derivatives
      call deriv (dudy, ucos,'y',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
      call deriv (dvdy, vcos,'y',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

!     Calculate cos correction factor
      do k=1,nz
        do j=1,ny
          do i=1,nx
               if (abs(dudy(i,j,k)-mdv).gt.eps) then
                  dudy(i,j,k)=dudy(i,j,k)/(coslat(i,j)+zerodiv)
               endif
               if (abs(dvdy(i,j,k)-mdv).gt.eps) then
                  dvdy(i,j,k)=dvdy(i,j,k)/(coslat(i,j)+zerodiv)
               endif
            enddo
         enddo
      enddo

!     Calculate x-derivatives
      call deriv (dudx, u,'x',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
      call deriv (dvdx, v,'x',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

!     Calculate deformations
      do k=1,nz
        do j=1,ny
          do i=1,nx
               if ((abs(dudx(i,j,k)-mdv).gt.eps).and. &
                (abs(dvdy(i,j,k)-mdv).gt.eps)) then
                  F1(i,j,k)=dudx(i,j,k) - dvdy(i,j,k)
               else
                  F1(i,j,k)=mdv
               endif
               if ((abs(dvdx(i,j,k)-mdv).gt.eps).and. &
                (abs(dudy(i,j,k)-mdv).gt.eps)) then
                  F2(i,j,k)=dvdx(i,j,k) + dudy(i,j,k)
               else
                  F2(i,j,k)=mdv
               endif
            enddo
         enddo
      enddo

      end subroutine deformation


!     ****************************************************************
!     * SUBROUTINE SECTION: GRID HANDLING                            *
!     ****************************************************************

!     -----------------------------------------------------------------
!     Create the aura around a 3d field
!     -----------------------------------------------------------------

      subroutine aura (gri,aur,dir,nx,ny,nz,xmin,ymin,dx,dy,mdv)

!     Create a one-point aura around the grid, in order to avoid nasty
!     problems when calculating fields at boundary points

      implicit none

!     Declaration of subroutine parameters
      integer nx,ny,nz
!f2py intent(in) nx,ny,nz
      real    gri(nx,ny,nz)
!f2py intent(in) gri
      real    aur(0:nx+1,0:ny+1,0:nz+1)
!f2py intent(out) aur
      integer dir
!f2py intent(in) dir
      real    xmin,ymin,dx,dy
!f2py intent(in) xmin,ymin,dx,dy
      real    mdv
!f2py intent(in) mdv

!     Numerical and physical parameters
      real       eps
      parameter  (eps=0.01)

!     Auxiliary variables
      integer i,j,k
      real    xmax,ymax
      integer domx,domy
      real    mean,count
      integer kmin,kmax

!     Set the z dimension for the output array
      if (nz.gt.1) then
         kmin=0
         kmax=nz+1
      elseif (nz.eq.1) then
         kmin=0
         kmax=2
      endif

!     Determine the x topology of the grid
!     1: periodic, not closed;
!     2: periodic, closed;
!     0: not periodic (and therefore not closed)
      xmax=xmin+real(nx-1)*dx
      ymax=ymin+real(ny-1)*dy
      if (abs(xmax-xmin-360.).lt.eps) then
         domx=2		! switched to 2 (Lukas Papritz)
      elseif (abs(xmax-xmin-360.+dx).lt.eps) then
         domx=1		! switched to 1 (Lukas Papritz)
      else
         domx=0
      endif

!     Determine the y topology of the grid
!     1    : neither north, nor south pole;
!     mod 2: exactly at south pole (closed)
!     mod 3: exactly at north pole (closed)
!     mod 5: one grid point north of south pole (not closed)
!     mod 7: one grid point south of north pole (not closed)
      domy=1
      if (abs(ymin+90.).lt.eps) then
         domy=2
      endif
      if (abs(ymax-90.).lt.eps) then
         domy=domy*3
      endif
      if (abs(ymin+90.-dy).lt.eps) then
         domy=domy*5
      endif
      if (abs(ymin-90.+dy).lt.eps) then
         domy=domy*7
      endif

!     Forward transformation (create aura)
      if (dir.eq.1) then

!        Copy the interior part
         aur(1:nx,1:ny,1:nz)=gri(1:nx,1:ny,1:nz)

!        Upper boundary
         if (nz.gt.1) then
            do i=1,nx
               do j=1,ny
                  if ((abs(aur(i,j,  nz)-mdv).gt.eps).and. &
                      (abs(aur(i,j,nz-1)-mdv).gt.eps)) then
                     aur(i,j,nz+1) = 2.*aur(i,j,nz) - aur(i,j,nz-1)
                  else
                     aur(i,j,nz+1) = mdv
                  endif
               enddo
            enddo
         else
            do i=1,nx
               do j=1,ny
                  aur(i,j,2)=aur(i,j,1)
               enddo
            enddo
         endif

!        Lower boundary
         if (nz.gt.1) then
            do i=1,nx
               do j=1,ny
                  if ((abs(aur(i,j,1)-mdv).gt.eps).and. &
                      (abs(aur(i,j,2)-mdv).gt.eps)) then
                     aur(i,j,0) = 2.*aur(i,j,1) - aur(i,j,2)
                  else
                     aur(i,j,0) = mdv
                  endif
               enddo
            enddo
         else
            do i=1,nx
               do j=1,ny
                  aur(i,j,0)=aur(i,j,1)
               enddo
            enddo
         endif

!        Northern and southern boundary, not near the poles
         if (mod(domy,1).eq.0) then
            do i=1,nx
               do k=kmin,kmax
                  if ((abs(aur(i,  ny,k)-mdv).gt.eps).and. &
                      (abs(aur(i,ny-1,k)-mdv).gt.eps)) then
                     aur(i,ny+1,k) = 2.*aur(i,ny,k)-aur(i,ny-1,k)
                  else
                     aur(i,ny+1,k) = mdv
                  endif
                  if  ((abs(aur(i,1,k)-mdv).gt.eps).and. &
                       (abs(aur(i,2,k)-mdv).gt.eps)) then
                     aur(i,0,k) = 2.*aur(i,1,k)-aur(i,2,k)
                  else
                     aur(i,0,k) = mdv
                  endif
               enddo
            enddo
         endif

!        Southern boundary, one grid point north of pole
!        Set the pole point to the mean of the nearest points
         if (mod(domy,5).eq.0) then
            do k=kmin,kmax
               mean=0.
               count=0.
               do i=1,nx
                  if (abs(aur(i,1,k)-mdv).gt.eps) then
                     mean=mean+aur(i,1,k)
                     count=count+1.
                  endif
               enddo
               if (count.gt.0.) then
                  mean=mean/count
               else
                  mean=mdv
               endif
               do i=1,nx
                  aur(i,0,k) = mean
               enddo
            enddo
         endif

!        Northern boundary, one grid point south of pole
!        Set the pole point to the mean of the nearest points
         if (mod(domy,7).eq.0) then
            do k=kmin,kmax
               mean=0.
               count=0.
               do i=1,nx
                  if (abs(aur(i,ny,k)-mdv).gt.eps) then
                     mean=mean+aur(i,ny,k)
                     count=count+1.
                  endif
               enddo
               if (count.gt.0.) then
                  mean=mean/count
               else
                  mean=mdv
               endif
               do i=1,nx
                  aur(i,ny+1,k) = mean
               enddo
            enddo
         endif

!        Southern boundary, exactly at south pole
         if (mod(domy,2).eq.0) then
            do i=1,nx
               do k=kmin,kmax
                  aur(i,0,k)=mdv
               enddo
            enddo
         endif

!        Northern boundary, exactly at north pole
         if (mod(domy,3).eq.0) then
            do i=1,nx
               do k=kmin,kmax
                  aur(i,ny+1,k)=mdv
               enddo
            enddo
         endif

!        The domain is periodic in x, but not closed
         if (domx.eq.1) then
            do j=0,ny+1
               do k=kmin,kmax
                  aur(   0,j,k) = aur(nx,j,k)
                  aur(nx+1,j,k) = aur( 1,j,k)
               enddo
            enddo
         endif

!        The domain is periodic in x and closed
         if (domx.eq.2) then
            do j=0,ny+1
               do k=kmin,kmax
                  aur(   0,j,k) = aur(nx-1,j,k)
                  aur(nx+1,j,k) = aur(   2,j,k)
               enddo
            enddo
         endif

!        The domain is not periodic in x
         if (domx.eq.0) then
            do j=0,ny+1
               do k=kmin,kmax
                  if ((abs(aur(1,j,k)-mdv).gt.eps).and. &
                      (abs(aur(2,j,k)-mdv).gt.eps)) then
                     aur(0,j,k) = 2.*aur(1,j,k) - aur(2,j,k)
                  else
                     aur(0,j,k) = mdv
                  endif
                  if ((abs(aur(  nx,j,k)-mdv).gt.eps).and. &
                      (abs(aur(nx-1,j,k)-mdv).gt.eps)) then
                     aur(nx+1,j,k) = 2.*aur(nx,j,k) - aur(nx-1,j,k)
                  else
                     aur(nx+1,j,k) = mdv
                  endif
               enddo
            enddo
         endif

      endif

!     Backward transformation
      if (dir.eq.-1) then

         if (nz.gt.1) then
            gri(1:nx,1:ny,1:nz)=aur(1:nx,1:ny,1:nz)
         elseif (nz.eq.1) then
            gri(1:nx,1:ny,1)=aur(1:nx,1:ny,1)
         endif

      endif

      end subroutine aura



!     -----------------------------------------------------------------
!     Horizontal and vertical derivatives for 3d fields
!     -----------------------------------------------------------------

      subroutine deriv (dfield,field,direction, &
                      vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

!     Calculate horizontal and vertical derivatives of the 3d field <field>.
!     The direction of the derivative is specified in <direction>
!         'x','y'          : Horizontal derivative in x and y direction
!         'p','z','t','m'  : Vertical derivative (pressure, height, theta, model)
!     The 3d field <vert> specifies the isosurfaces along which the horizontal
!     derivatives are calculated or the levels for the vertical derivatives. If
!     horizontal derivatives along model levels should be calculated, pass an
!     index arrray <vert(i,j,k)=k>.

      implicit none

!     Input and output parameters
      integer    nx,ny,nz
!f2py intent(in) nx,ny,nz
      real       dfield(nx,ny,nz)
!f2py intent(out) dfield
      real       field(nx,ny,nz)
!f2py intent(in) field
      real       vert(nx,ny,nz)
!f2py intent(in) vert
      character  direction
!f2py intent(in) direction
      real       xmin,ymin,dx,dy
!f2py intent(in) xmin,ymin,dx,dy
      real       mdv
!f2py intent(in) mdv

!     Numerical and physical parameters
      real       pi180
      parameter  (pi180=3.141592654/180.)
      real       deltay
      parameter  (deltay=111.1775E3)
      real       zerodiv
      parameter  (zerodiv=0.00000001)
      real       eps
      parameter  (eps=0.01)

!     Auxiliary variables
      integer    i,j,k
      real       vmin,vmax
      real       scale,lat
      real       vu,vl,vuvl,vlvu
      real       df(0:nx+1,0:ny+1,0:nz+1)
      real       f(0:nx+1,0:ny+1,0:nz+1)
      real       v(0:nx+1,0:ny+1,0:nz+1)

!     Create the aura around the grid for fast boundary handling
      call aura (field,f,1,nx,ny,nz,xmin,ymin,dx,dy,mdv)
      call aura (vert, v,1,nx,ny,nz,xmin,ymin,dx,dy,mdv)

!     Vertical derivative
      if ((direction.eq.'z').or. &
          (direction.eq.'th').or. &
          (direction.eq.'p').or. &
          (direction.eq.'m').and. &
          (nz.gt.1)) then

!        Finite differencing
         do i=1,nx
            do j=1,ny
               do k=1,nz

                  if ((abs(f(i,j,k+1)-mdv).gt.eps).and. &
                      (abs(f(i,j,k-1)-mdv).gt.eps).and. &
                      (abs(v(i,j,k+1)-mdv).gt.eps).and. &
                      (abs(v(i,j,k-1)-mdv).gt.eps).and. &
                      (abs(v(i,j,  k)-mdv).gt.eps)) then

                     vu = v(i,j,  k)-v(i,j,k+1)
                     vl = v(i,j,k-1)-v(i,j,  k)
                     vuvl = vu/(vl+zerodiv)
                     vlvu = 1./(vuvl+zerodiv)

                     df(i,j,k) = 1./(vu+vl) &
                                * (vuvl*(f(i,j,k-1)-f(i,j,  k))  &
                                +  vlvu*(f(i,j,  k)-f(i,j,k+1)))

                  else
                     df(i,j,k) = mdv
                  endif

               enddo
            enddo
         enddo

!     Horizontal derivative in the y direction: 3d
      elseif ((direction.eq.'y').and.(nz.gt.1)) then

!        Scale factor for derivative in m
         scale=1./(2.*dy*deltay)

!        Finite differencing
         do i=1,nx
            do j=1,ny
               do k=1,nz

                 if ((abs(f(i,j+1,k)-mdv).gt.eps).and. &
                     (abs(f(i,j-1,k)-mdv).gt.eps).and. &
                     (abs(f(i,j,k+1)-mdv).gt.eps).and. &
                     (abs(f(i,j,k-1)-mdv).gt.eps).and. &
                     (abs(v(i,j,k+1)-mdv).gt.eps).and. &
                     (abs(v(i,j,k+1)-mdv).gt.eps).and. &
                     (abs(v(i,j,k-1)-mdv).gt.eps).and. &
                     (abs(v(i,j,k-1)-mdv).gt.eps).and. &
                     (abs(v(i,j+1,k)-mdv).gt.eps).and. &
                     (abs(v(i,j+1,k)-mdv).gt.eps).and. &
                     (abs(v(i,j-1,k)-mdv).gt.eps).and. &
                     (abs(v(i,j-1,k)-mdv).gt.eps)) then

                     df(i,j,k) = &
                         scale*(f(i,j+1,k)-f(i,j-1,k)) &
                        -(f(i,j,k+1)-f(i,j,k-1))/(v(i,j,k+1)-v(i,j,k-1)) &
                        *scale*(v(i,j+1,k)-v(i,j-1,k))

                  else
                     df(i,j,k) = mdv
                  endif

               enddo
            enddo
         enddo

!     Horizontal derivative in the x direction: 3d
      elseif ((direction.eq.'x').and.(nz.gt.1)) then

!        Finite differencing
         do j=1,ny

!           Scale factor for derivatives in m (latitude dependent)
            lat=ymin+real(j-1)*dy
            scale=1./(2.*dx*deltay*cos(pi180*lat)+zerodiv)

            do i=1,nx
               do k=1,nz

                 if ((abs(f(i+1,j,k)-mdv).gt.eps).and. &
                     (abs(f(i-1,j,k)-mdv).gt.eps).and. &
                     (abs(f(i,j,k+1)-mdv).gt.eps).and. &
                     (abs(f(i,j,k-1)-mdv).gt.eps).and. &
                     (abs(v(i,j,k+1)-mdv).gt.eps).and. &
                     (abs(v(i,j,k+1)-mdv).gt.eps).and. &
                     (abs(v(i,j,k-1)-mdv).gt.eps).and. &
                     (abs(v(i,j,k-1)-mdv).gt.eps).and. &
                     (abs(v(i+1,j,k)-mdv).gt.eps).and. &
                     (abs(v(i+1,j,k)-mdv).gt.eps).and. &
                     (abs(v(i-1,j,k)-mdv).gt.eps).and. &
                     (abs(v(i-1,j,k)-mdv).gt.eps)) then

                     df(i,j,k) = &
                         scale*(f(i+1,j,k)-f(i-1,j,k)) &
                        -(f(i,j,k+1)-f(i,j,k-1))/(v(i,j,k+1)-v(i,j,k-1)) &
                        *scale*(v(i+1,j,k)-v(i-1,j,k))

                  else
                     df(i,j,k) = mdv
                  endif

               enddo
            enddo
         enddo

!     Horizontal derivative in the y direction: 2d
      elseif ((direction.eq.'y').and.(nz.eq.1)) then

!        Scale factor for derivative in m
         scale=1./(2.*dy*deltay)

!        Finite differencing
         do i=1,nx
            do j=1,ny

               if ((abs(f(i,j+1,1)-mdv).gt.eps).and. &
                   (abs(f(i,j-1,1)-mdv).gt.eps)) then

                  df(i,j,1) = scale*(f(i,j+1,1)-f(i,j-1,1))

               else
                  df(i,j,1) = mdv
               endif

            enddo
         enddo

!     Horizontal derivative in the x direction: 2d
      elseif ((direction.eq.'x').and.(nz.eq.1)) then

!        Finite differencing
         do j=1,ny

!           Scale factor for derivatives in m (latitude dependent)
            lat=ymin+real(j-1)*dy
            scale=1./(2.*dx*deltay*cos(pi180*lat)+zerodiv)

            do i=1,nx

               if ((abs(f(i+1,j,1)-mdv).gt.eps).and. &
                   (abs(f(i-1,j,1)-mdv).gt.eps)) then

                  df(i,j,1) = scale*(f(i+1,j,1)-f(i-1,j,1))

               else
                  df(i,j,1) = mdv
               endif

            enddo
         enddo

!     Undefined direction for derivative
      else

         print*,'Invalid direction of derivative... Stop'
         stop

      endif

!     Get rid of the aura
      call aura (dfield,df,-1,nx,ny,nz,xmin,ymin,dx,dy,mdv)

      end subroutine deriv


!     -----------------------------------------------------------------
!     Time derivative on pressure levels
!     -----------------------------------------------------------------

      subroutine deriv_time (dAnow,Anew,Anow,Aold,Pnew,Pnow,Pold,dt, &
                             xmin,ymin,dx,dy,nx,ny,nz,mdv)

!     Calculate time derivative of 3d field <A> at time now.
!     <Anew>, <Anow> and <Aold> (<Pnew>, <Pnow> and <Pold>) is the field A (pressure) at time
!     now + dt, now, now-dt
!
!     [dt] = hours

!     The 3d field <P> specifies the pressure on model levels.

      implicit none

!     Input and output parameters
      integer    nx,ny,nz
!f2py intent(in) nx,ny,nz
      real       Anew(nx,ny,nz)
!f2py intent(in) Anew
      real       Anow(nx,ny,nz)
!f2py intent(in) Anow
      real       Aold(nx,ny,nz)
!f2py intent(in) Aold
      real       Pnew(nx,ny,nz)
!f2py intent(in) Pnew
      real       Pnow(nx,ny,nz)
!f2py intent(in) Pnow
      real       Pold(nx,ny,nz)
!f2py intent(in) Pold
      real       xmin,ymin,dx,dy
!f2py intent(in) xmin,ymin,dx,dy
      real       mdv
!f2py intent(in) mdv
      real 	 dt
!f2py intent(in) dt

!     Numerical and physical parameters
      real       zerodiv
      parameter  (zerodiv=0.00000001)
      real       eps
      parameter  (eps=0.01)

!     Auxiliary variables
      integer    i,j,k
      real       scale
      real       dAnow(nx,ny,nz)
!f2py intent(out) dAnow
      real       Anow_aura(0:nx+1,0:ny+1,0:nz+1)
      real       Pnow_aura(0:nx+1,0:ny+1,0:nz+1)

!     Create the aura around the grid for fast boundary handling
      call aura (Anow,Anow_aura,1,nx,ny,nz,xmin,ymin,dx,dy,mdv)
      call aura (Pnow,Pnow_aura,1,nx,ny,nz,xmin,ymin,dx,dy,mdv)


!      Scale factor [1/s]
       scale=1/(2. * dt * 3600.)

!      Finite differencing
       do i=1,nx
          do j=1,ny
             do k=1,nz
                 if ((abs(Anew(i,j,k)-mdv).gt.eps).and. &
                     (abs(Aold(i,j,k)-mdv).gt.eps).and. &
                     (abs(Anow_aura(i,j,k+1)-mdv).gt.eps).and. &
                     (abs(Anow_aura(i,j,k-1)-mdv).gt.eps).and. &
                     (abs(Pnow_aura(i,j,k+1)-mdv).gt.eps).and. &
                     (abs(Pnow_aura(i,j,k-1)-mdv).gt.eps).and. &
                     (abs(Pnew(i,j,k)-mdv).gt.eps).and. &
                     (abs(Pnow(i,j,k)-mdv).gt.eps)) then

                     dAnow(i,j,k) = &
                         scale*(Anew(i,j,k)-Aold(i,j,k)) &
                        -(Anow_aura(i,j,k+1)-Anow_aura(i,j,k-1))/(Pnow_aura(i,j,k+1)-Pnow_aura(i,j,k-1)) &
                        *scale*(Pnew(i,j,k)-Pold(i,j,k))   ! Seb: corrected from Pnow

                  else
                     dAnow(i,j,k) = mdv
                  endif

             enddo
          enddo
       enddo

      end subroutine deriv_time

!     ****************************************************************

!     ****************************************************************

! SR end module deriv
