c     ****************************************************************
c     * SUBROUTINE SECTION: ELEMENTARY VECTOR ANALYTICAL EXPRESSIONS *
c     ****************************************************************

c     -----------------------------------------------------------------
c     Gradient
c     -----------------------------------------------------------------

      subroutine grad(gradx,grady,scalar,
     >                vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

c     Calculate the two vector components <gradx> and <grady> of the
c     scalar field <scalar>.  The vertical coordinate is sepcified in
c     <vert>, the grid in <xmin,ymin,dx,dy,nx,ny,nz,mdv>.

      implicit none

c     Declaration of subroutine parameters
      integer nx,ny,nz
cf2py intent(in) nx,ny,nz
      real    gradx(nx,ny,nz)
cf2py intent(out) gradx
      real    grady(nx,ny,nz)
cf2py intent(out) grady
      real    scalar(nx,ny,nz)
cf2py intent(in) scalar
      real    vert(nx,ny,nz)
cf2py intent(in) vert
      real    xmin,ymin,dx,dy
cf2py intent(in) xmin,ymin,dx,dy
      real    mdv
cf2py intent(in) mdv

c     Calculate the derivatives in x and y direction
      call deriv (gradx,scalar,'x',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
      call deriv (grady,scalar,'y',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

      end


c     -----------------------------------------------------------------
c     Rotation
c     -----------------------------------------------------------------

      subroutine rot(vort3,u3,v3,
     >               vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

c     Calculate the vertical component <vort> of the vector field <uu>
c     <vv>. The vertical coordinate is sepcified in <vert>, the grid
c     in <xmin,ymin,dx,dy,nx,ny,nz,mdv>.

      use omp_lib

      implicit none

c     Declaration of subroutine parameters
      integer    nx,ny,nz
cf2py intent(in) nx,ny,nz
      real       u3(nx,ny,nz)
cf2py intent(in) u3
      real       v3(nx,ny,nz)
cf2py intent(in) v3
      real       vort3(nx,ny,nz)
cf2py intent(out) vort3
      real       vert(nx,ny,nz)
cf2py intent(in) vert
      real       xmin,ymin,dx,dy
cf2py intent(in) xmin,ymin,dx,dy
      real       mdv
cf2py intent(in) mdv

c     Mathematical and physical parameters
      real       pi180
      parameter  (pi180=3.141592654/180.)
      real       zerodiv
      parameter  (zerodiv=0.00000001)
      real       eps
      parameter  (eps=0.01)

c     Auxiliary variables
      integer    i,j,k
      real       ucos(nx,ny,nz)
      real       coslat(nx,ny)
      real       dvdx(nx,ny,nz)
      real       dudy(nx,ny,nz)
      real       lat

c     Calculate Cos factor: correction for spherical coordinates
!$omp parallel do private(lat)
      do i=1,nx
         do j=1,ny
            lat=ymin+real(j-1)*dy
            coslat(i,j)=cos(pi180*lat)
         enddo
      enddo
!$omp end parallel do

c     Derivatives of zonal velocity U: correction for spherical coordinates
!$omp parallel do
      do i=1,nx
         do j=1,ny
            do k=1,nz
               if(abs(u3(i,j,k)-mdv).gt.eps) then
                  ucos(i,j,k)=u3(i,j,k)*coslat(i,j)
               endif
            enddo
         enddo
      enddo
!$omp end parallel do
      call deriv (dudy, ucos,'y',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
!$omp parallel do
      do i=1,nx
         do j=1,ny
            do k=1,nz
               if (abs(dudy(i,j,k)-mdv).gt.eps) then
                  dudy(i,j,k)=dudy(i,j,k)/(coslat(i,j)+zerodiv)
               endif
            enddo
         enddo
      enddo
!$omp end parallel do

c     Derivatives of meridional velocity V
      call deriv (dvdx, v3,'x',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

c     Calculate rotation of vector field
!$omp parallel do
      do i=1,nx
         do j=1,ny
            do k=1,nz
               if ((abs(dvdx(i,j,k)-mdv).gt.eps).and.
     >             (abs(dudy(i,j,k)-mdv).gt.eps)) then
                  vort3(i,j,k)=dvdx(i,j,k)-dudy(i,j,k)
               else
                  vort3(i,j,k)=mdv
               endif
            enddo
         enddo
      enddo
!$omp end parallel do

      end

c     -----------------------------------------------------------------
c     Divergence
c     -----------------------------------------------------------------

      subroutine div(div3,u3,v3,
     >               vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

c     Calculate the divergence <div> of the horizontal vector field <uu>
c     <vv>. The vertical coordinate is sepcified in <vert>, the grid
c     in <xmin,ymin,dx,dy,nx,ny,nz,mdv>.

      use omp_lib

      implicit none

c     Declaration of subroutine parameters
      integer    nx,ny,nz
cf2py intent(in) nx,ny,nz
      real       u3(nx,ny,nz)
cf2py intent(in) u3
      real       v3(nx,ny,nz)
cf2py intent(in) v3
      real       div3(nx,ny,nz)
cf2py intent(out) div3
      real       vert(nx,ny,nz)
cf2py intent(in) vert
      real       xmin,ymin,dx,dy
cf2py intent(in) xmin,ymin,dx,dy
      real       mdv
cf2py intent(in) mdv

c     Mathematical and physical parameters
      real       pi180
      parameter  (pi180=3.141592654/180.)
      real       zerodiv
      parameter  (zerodiv=0.00000001)
      real       eps
      parameter  (eps=0.01)

c     Auxiliary variables
      integer    i,j,k
      real       vcos(nx,ny,nz)
      real       coslat(nx,ny)
      real       dvdy(nx,ny,nz)
      real       dudx(nx,ny,nz)
      real       lat

c     Calculate Cos factor: correction for spherical coordinates
!$omp parallel do private(lat)
      do i=1,nx
         do j=1,ny
            lat=ymin+real(j-1)*dy
            coslat(i,j)=cos(pi180*lat)
         enddo
      enddo
!$omp end parallel do

c     Derivatives of zonal velocity U: correction for spherical coordinates
!$omp parallel do
      do i=1,nx
         do j=1,ny
            do k=1,nz
               if(abs(v3(i,j,k)-mdv).gt.eps) then
                  vcos(i,j,k)=v3(i,j,k)*coslat(i,j)
               endif
            enddo
         enddo
      enddo
!$omp end parallel do
      call deriv (dvdy, vcos,'y',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
!$omp parallel do
      do i=1,nx
         do j=1,ny
            do k=1,nz
               if (abs(dvdy(i,j,k)-mdv).gt.eps) then
                  dvdy(i,j,k)=dvdy(i,j,k)/(coslat(i,j)+zerodiv)
               endif
            enddo
         enddo
      enddo
!$omp end parallel do

c     Derivatives of meridional velocity V
      call deriv (dudx, u3,'x',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

c     Calculate divergence of vector field
!$omp parallel do
      do i=1,nx
         do j=1,ny
            do k=1,nz
               if ((abs(dvdy(i,j,k)-mdv).gt.eps).and.
     >             (abs(dudx(i,j,k)-mdv).gt.eps)) then
                  div3(i,j,k)=dvdy(i,j,k)+dudx(i,j,k)
               else
                  div3(i,j,k)=mdv
               endif
            enddo
         enddo
      enddo
!$omp end parallel do

      end


c     ****************************************************************
c     * SUBROUTINE SECTION: GRID HANDLING                            *
c     ****************************************************************

c     -----------------------------------------------------------------
c     Create the aura around a 3d field
c     -----------------------------------------------------------------

      subroutine aura (gri,aur,dir,nx,ny,nz,xmin,ymin,dx,dy,mdv)

c     Create a one-point aura around the grid, in order to avoid nasty
c     problems when calculating fields at boundary points

      use omp_lib

      implicit none

c     Declaration of subroutine parameters
      integer nx,ny,nz
cf2py intent(in) nx,ny,nz
      real    gri(nx,ny,nz)
cf2py intent(in) gri
      real    aur(0:nx+1,0:ny+1,0:nz+1)
cf2py intent(out) aur
      integer dir
cf2py intent(in) dir
      real    xmin,ymin,dx,dy
cf2py intent(in) xmin,ymin,dx,dy
      real    mdv
cf2py intent(in) mdv

c     Numerical and physical parameters
      real       eps
      parameter  (eps=0.01)

c     Auxiliary variables
      integer i,j,k
      real    xmax,ymax
      integer domx,domy
      real    mean,count
      integer kmin,kmax

c     Set the z dimension for the output array
      if (nz.gt.1) then
         kmin=0
         kmax=nz+1
      elseif (nz.eq.1) then
         kmin=0
         kmax=2
      endif

c     Determine the x topology of the grid
c     1: periodic, not closed;
c     2: periodic, closed;
c     0: not periodic (and therefore not closed)
      xmax=xmin+real(nx-1)*dx
      ymax=ymin+real(ny-1)*dy
      if (abs(xmax-xmin-360.).lt.eps) then
         domx=1
      elseif (abs(xmax-xmin-360.+dx).lt.eps) then
         domx=2
      else
         domx=0
      endif

c     Determine the y topology of the grid
c     1    : neither north, nor south pole;
c     mod 2: exactly at south pole (closed)
c     mod 3: exactly at north pole (closed)
c     mod 5: one grid point north of south pole (not closed)
c     mod 7: one grid point south of north pole (not closed)
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

c     Forward transformation (create aura)
      if (dir.eq.1) then

c        Copy the interior part
         aur(1:nx,1:ny,1:nz)=gri(1:nx,1:ny,1:nz)

c        Upper boundary
         if (nz.gt.1) then
!$omp parallel do
            do i=1,nx
               do j=1,ny
                  if ((abs(aur(i,j,  nz)-mdv).gt.eps).and.
     >                (abs(aur(i,j,nz-1)-mdv).gt.eps)) then
                     aur(i,j,nz+1) = 2.*aur(i,j,nz) - aur(i,j,nz-1)
                  else
                     aur(i,j,nz+1) = mdv
                  endif
               enddo
            enddo
!$omp end parallel do
         else
!$omp parallel do
            do i=1,nx
               do j=1,ny
                  aur(i,j,2)=aur(i,j,1)
               enddo
            enddo
!$omp end parallel do
         endif

c        Lower boundary
         if (nz.gt.1) then
!$omp parallel do
            do i=1,nx
               do j=1,ny
                  if ((abs(aur(i,j,1)-mdv).gt.eps).and.
     >                (abs(aur(i,j,2)-mdv).gt.eps)) then
                     aur(i,j,0) = 2.*aur(i,j,1) - aur(i,j,2)
                  else
                     aur(i,j,0) = mdv
                  endif
               enddo
            enddo
!$omp end parallel do
         else
!$omp parallel do
            do i=1,nx
               do j=1,ny
                  aur(i,j,0)=aur(i,j,1)
               enddo
            enddo
!$omp end parallel do
         endif

c        Northern and southern boundary, not near the poles
         if (mod(domy,1).eq.0) then
!$omp parallel do
            do i=1,nx
               do k=kmin,kmax
                  if ((abs(aur(i,  ny,k)-mdv).gt.eps).and.
     >                (abs(aur(i,ny-1,k)-mdv).gt.eps)) then
                     aur(i,ny+1,k) = 2.*aur(i,ny,k)-aur(i,ny-1,k)
                  else
                     aur(i,ny+1,k) = mdv
                  endif
                  if  ((abs(aur(i,1,k)-mdv).gt.eps).and.
     >                 (abs(aur(i,2,k)-mdv).gt.eps)) then
                     aur(i,0,k) = 2.*aur(i,1,k)-aur(i,2,k)
                  else
                     aur(i,0,k) = mdv
                  endif
               enddo
            enddo
!$omp end parallel do
         endif

c        Southern boundary, one grid point north of pole
c        Set the pole point to the mean of the nearest points
         if (mod(domy,5).eq.0) then
!$omp parallel do private(mean, count)
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
!$omp end parallel do
         endif

c        Northern boundary, one grid point south of pole
c        Set the pole point to the mean of the nearest points
         if (mod(domy,7).eq.0) then
!$omp parallel do private(mean, count)
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
!$omp end parallel do
         endif

c        Southern boundary, exactly at south pole
         if (mod(domy,2).eq.0) then
!$omp parallel do
            do i=1,nx
               do k=kmin,kmax
                  aur(i,0,k)=mdv
               enddo
            enddo
!$omp end parallel do
         endif

c        Northern boundary, exactly at north pole
         if (mod(domy,3).eq.0) then
!$omp parallel do
            do i=1,nx
               do k=kmin,kmax
                  aur(i,ny+1,k)=mdv
               enddo
            enddo
!$omp end parallel do
         endif

c        The domain is periodic in x, but not closed
         if (domx.eq.1) then
!$omp parallel do
            do j=0,ny+1
               do k=kmin,kmax
                  aur(   0,j,k) = aur(nx,j,k)
                  aur(nx+1,j,k) = aur( 1,j,k)
               enddo
            enddo
!$omp end parallel do
         endif

c        The domain is periodic in x and closed
         if (domx.eq.2) then
!$omp parallel do
            do j=0,ny+1
               do k=kmin,kmax
                  aur(   0,j,k) = aur(nx-1,j,k)
                  aur(nx+1,j,k) = aur(   2,j,k)
               enddo
            enddo
!$omp end parallel do
         endif

c        The domain is not periodic in x
         if (domx.eq.0) then
!$omp parallel do
            do j=0,ny+1
               do k=kmin,kmax
                  if ((abs(aur(1,j,k)-mdv).gt.eps).and.
     >                (abs(aur(2,j,k)-mdv).gt.eps)) then
                     aur(0,j,k) = 2.*aur(1,j,k) - aur(2,j,k)
                  else
                     aur(0,j,k) = mdv
                  endif
                  if ((abs(aur(  nx,j,k)-mdv).gt.eps).and.
     >                (abs(aur(nx-1,j,k)-mdv).gt.eps)) then
                     aur(nx+1,j,k) = 2.*aur(nx,j,k) - aur(nx-1,j,k)
                  else
                     aur(nx+1,j,k) = mdv
                  endif
               enddo
            enddo
!$omp end parallel do
         endif

      endif

c     Backward transformation
      if (dir.eq.-1) then

         if (nz.gt.1) then
            gri(1:nx,1:ny,1:nz)=aur(1:nx,1:ny,1:nz)
         elseif (nz.eq.1) then
            gri(1:nx,1:ny,1)=aur(1:nx,1:ny,0)
         endif

      endif

      end



c     -----------------------------------------------------------------
c     Horizontal and vertical derivatives for 3d fields
c     -----------------------------------------------------------------

      subroutine deriv (dfield,field,direction,
     >                  vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

c     Calculate horizontal and vertical derivatives of the 3d field <field>.
c     The direction of the derivative is specified in <direction>
c         'x','y'          : Horizontal derivative in x and y direction
c         'p','z','t','m'  : Vertical derivative (pressure, height, theta, model)
c     The 3d field <vert> specifies the isosurfaces along which the horizontal
c     derivatives are calculated or the levels for the vertical derivatives. If
c     horizontal derivatives along model levels should be calculated, pass an
c     index arrray <vert(i,j,k)=k>.

      use omp_lib

      implicit none

c     Input and output parameters
      integer    nx,ny,nz
cf2py intent(in) nx,ny,nz
      real       dfield(nx,ny,nz)
cf2py intent(out) dfield
      real       field(nx,ny,nz)
cf2py intent(in) field
      real       vert(nx,ny,nz)
cf2py intent(in) vert
      character  direction
cf2py intent(in) direction
      real       xmin,ymin,dx,dy
cf2py intent(in) xmin,ymin,dx,dy
      real       mdv
cf2py intent(in) mdv

c     Numerical and physical parameters
      real       pi180
      parameter  (pi180=3.141592654/180.)
      real       deltay
      parameter  (deltay=111.1775E3)
      real       zerodiv
      parameter  (zerodiv=0.00000001)
      real       eps
      parameter  (eps=0.01)

c     Auxiliary variables
      integer    i,j,k
      real       vmin,vmax
      real       scale,lat
      real       vu,vl,vuvl,vlvu
      real       df(0:nx+1,0:ny+1,0:nz+1)
      real       f(0:nx+1,0:ny+1,0:nz+1)
      real       v(0:nx+1,0:ny+1,0:nz+1)

c     Create the aura around the grid for fast boundary handling
      call aura (field,f,1,nx,ny,nz,xmin,ymin,dx,dy,mdv)
      call aura (vert, v,1,nx,ny,nz,xmin,ymin,dx,dy,mdv)

c     Vertical derivative
      if ((direction.eq.'z').or.
     >    (direction.eq.'th').or.
     >    (direction.eq.'p').or.
     >    (direction.eq.'m').and.
     >    (nz.gt.1)) then

c        Finite differencing
!$omp parallel do private(vu, vl, vuvl, vlvu)
         do i=1,nx
            do j=1,ny
               do k=1,nz

                  if ((abs(f(i,j,k+1)-mdv).gt.eps).and.
     >                (abs(f(i,j,k-1)-mdv).gt.eps).and.
     >                (abs(v(i,j,k+1)-mdv).gt.eps).and.
     >                (abs(v(i,j,k-1)-mdv).gt.eps).and.
     >                (abs(v(i,j,  k)-mdv).gt.eps)) then

                     vu = v(i,j,  k)-v(i,j,k+1)
                     vl = v(i,j,k-1)-v(i,j,  k)
                     vuvl = vu/(vl+zerodiv)
                     vlvu = 1./(vuvl+zerodiv)

                     df(i,j,k) = 1./(vu+vl)
     >                           * (vuvl*(f(i,j,k-1)-f(i,j,  k))
     >                           +  vlvu*(f(i,j,  k)-f(i,j,k+1)))

                  else
                     df(i,j,k) = mdv
                  endif

               enddo
            enddo
         enddo
!$omp end parallel do

c     Horizontal derivative in the y direction: 3d
      elseif ((direction.eq.'y').and.(nz.gt.1)) then

c        Scale factor for derivative in m
         scale=1./(2.*dy*deltay)

c        Finite differencing
!$omp parallel do
         do i=1,nx
            do j=1,ny
               do k=1,nz

                 if ((abs(f(i,j+1,k)-mdv).gt.eps).and.
     >               (abs(f(i,j-1,k)-mdv).gt.eps).and.
     >               (abs(f(i,j,k+1)-mdv).gt.eps).and.
     >               (abs(f(i,j,k-1)-mdv).gt.eps).and.
     >               (abs(v(i,j,k+1)-mdv).gt.eps).and.
     >               (abs(v(i,j,k+1)-mdv).gt.eps).and.
     >               (abs(v(i,j,k-1)-mdv).gt.eps).and.
     >               (abs(v(i,j,k-1)-mdv).gt.eps).and.
     >               (abs(v(i,j+1,k)-mdv).gt.eps).and.
     >               (abs(v(i,j+1,k)-mdv).gt.eps).and.
     >               (abs(v(i,j-1,k)-mdv).gt.eps).and.
     >               (abs(v(i,j-1,k)-mdv).gt.eps)) then

                     df(i,j,k) =
     >                   scale*(f(i,j+1,k)-f(i,j-1,k))
     >                  -(f(i,j,k+1)-f(i,j,k-1))/(v(i,j,k+1)-v(i,j,k-1))
     >                  *scale*(v(i,j+1,k)-v(i,j-1,k))

                  else
                     df(i,j,k) = mdv
                  endif

               enddo
            enddo
         enddo
!$omp end parallel do

c     Horizontal derivative in the x direction: 3d
      elseif ((direction.eq.'x').and.(nz.gt.1)) then

c        Finite differencing
!$omp parallel do private(scale,lat)
         do j=1,ny

c           Scale factor for derivatives in m (latitude dependent)
            lat=ymin+real(j-1)*dy
            scale=1./(2.*dx*deltay*cos(pi180*lat)+zerodiv)

            do i=1,nx
               do k=1,nz

                 if ((abs(f(i+1,j,k)-mdv).gt.eps).and.
     >               (abs(f(i-1,j,k)-mdv).gt.eps).and.
     >               (abs(f(i,j,k+1)-mdv).gt.eps).and.
     >               (abs(f(i,j,k-1)-mdv).gt.eps).and.
     >               (abs(v(i,j,k+1)-mdv).gt.eps).and.
     >               (abs(v(i,j,k+1)-mdv).gt.eps).and.
     >               (abs(v(i,j,k-1)-mdv).gt.eps).and.
     >               (abs(v(i,j,k-1)-mdv).gt.eps).and.
     >               (abs(v(i+1,j,k)-mdv).gt.eps).and.
     >               (abs(v(i+1,j,k)-mdv).gt.eps).and.
     >               (abs(v(i-1,j,k)-mdv).gt.eps).and.
     >               (abs(v(i-1,j,k)-mdv).gt.eps)) then

                     df(i,j,k) =
     >                   scale*(f(i+1,j,k)-f(i-1,j,k))
     >                  -(f(i,j,k+1)-f(i,j,k-1))/(v(i,j,k+1)-v(i,j,k-1))
     >                  *scale*(v(i+1,j,k)-v(i-1,j,k))

                  else
                     df(i,j,k) = mdv
                  endif

               enddo
            enddo
         enddo
!$omp end parallel do

c     Horizontal derivative in the y direction: 2d
      elseif ((direction.eq.'y').and.(nz.eq.1)) then

c        Scale factor for derivative in m
         scale=1./(2.*dy*deltay)

c        Finite differencing
!$omp parallel do
         do i=1,nx
            do j=1,ny

c SR  Add missing k index following deriv.f90 (gfortran compilation error)
c SR           if ((abs(f(i,j+1)-mdv).gt.eps).and.
c SR >             (abs(f(i,j-1)-mdv).gt.eps)) then
               if ((abs(f(i,j+1,1)-mdv).gt.eps).and.
     >             (abs(f(i,j-1,1)-mdv).gt.eps)) then

c SR              df(i,j) = scale*(f(i,j+1)-f(i,j-1))
                  df(i,j,1) = scale*(f(i,j+1,1)-f(i,j-1,1))

               else
c SR              df(i,j) = mdv
                  df(i,j,1) = mdv
               endif

            enddo
         enddo
!$omp end parallel do

c     Horizontal derivative in the x direction: 2d
      elseif ((direction.eq.'x').and.(nz.eq.1)) then

c        Finite differencing
!$omp parallel do private(scale, lat)
         do j=1,ny

c           Scale factor for derivatives in m (latitude dependent)
            lat=ymin+real(j-1)*dy
            scale=1./(2.*dx*deltay*cos(pi180*lat)+zerodiv)

            do i=1,nx

c SR  Add missing k index following deriv.f90 (gfortran compilation error)
c SR           if ((abs(f(i+1,j)-mdv).gt.eps).and.
c SR >             (abs(f(i-1,j)-mdv).gt.eps)) then
               if ((abs(f(i+1,j,1)-mdv).gt.eps).and.
     >             (abs(f(i-1,j,1)-mdv).gt.eps)) then

c SR              df(i,j) = scale*(f(i+1,j)-f(i-1,j))
                  df(i,j,1) = scale*(f(i+1,j,1)-f(i-1,j,1))

               else
c SR              df(i,j) = mdv
                  df(i,j,1) = mdv
               endif

            enddo
         enddo
!$omp end parallel do


c     Undefined direction for derivative
      else

         print*,'Invalid direction of derivative... Stop'
         stop

      endif

c     Get rid of the aura
      call aura (dfield,df,-1,nx,ny,nz,xmin,ymin,dx,dy,mdv)

      end

c     -----------------------------------------------------------------
c     Grid check
c     -----------------------------------------------------------------

      subroutine gridok (vert,vertcoord,xmin,ymin,dx,dy,nx,ny,nz,mdv)

c     Check whether the vertical coordinate field <vert> is ok. Depending
c     on the type of vertical coordinate a simple check is performed. The
c     allowed types are: <vertcoord='p','t','z','m') for pressure, theta,
c     geometrical height and model. The grid is specified by <xmin,ymin,dx,dy>.
c     If a coordinate point is not accepted, it is set to <mdv>.

      implicit none

c     Declaration of subroutine parameters
      integer   nx,ny,nz
cf2py intent(in) nx,ny,nz
      real      vert(0:nx+1,0:ny+1,0:nz+1)
cf2py intent(inout) vert
      real      xmin,ymin,dx,dy
cf2py intent(in) xmin,ymin,dx,dy
      real      mdv
cf2py intent(in) mdv
      character vertcoord
cf2py intent(in) vertcoord

c     Set the allowed values for the vertical coordinates
      real      pmin,pmax
      parameter (pmin=0.,pmax=120000.)
      real      tmin,tmax
      parameter (tmin=100.,tmax=1000.)
      real      zmin,zmax
      parameter (zmin=-200.,zmax=50000.)

c     Auxiliyr variables
      integer   i,j,k
      real      vmin,vmax

c     Set the allowed threshold
      if (vertcoord.eq.'p') then
         vmin=pmin
         vmax=pmax
      elseif (vertcoord.eq.'z') then
         vmin=zmin
         vmax=zmax
      elseif (vertcoord.eq.'t') then
         vmin=tmin
         vmax=tmax
      elseif (vertcoord.eq.'m') then
         vmin=0.
         vmax=real(nz+1)
      else
         print*,'Invalid vertical coordinate... Stop'
         stop
      endif

c     Loop over the grid
!$omp parallel do
      do i=0,nx+1
         do j=0,ny+1
            do k=0,nz+1

               if ((vert(i,j,k).lt.vmin).or.(vert(i,j,k).gt.vmax)) then
                  vert(i,j,k)=mdv
               endif

            enddo
         enddo
      enddo
!$omp end parallel do

      end
