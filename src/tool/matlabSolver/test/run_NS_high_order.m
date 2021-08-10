function [QoI, Obs] = run_NS_high_order(m)
    format long

    global dy nu Kx Ky K2 Lx Ly Nx Ny Nxy X Y K2inv AliasCor

%     m = [1.0, 0.5, 0.3, 0.8];
%     m = [1.0];

    nu = 0.1;

    Lx = 0.5;
    Ly = 0.5;  

    %%% Numerical parameters:
    Nx = 64;   % number of Fourier modes in discrete solution x-dir
    Ny = 64;   % number of Fourier modes in discrete solution y-dir
    Nxy = (Nx+1)*(Ny+1);    

    t = 0.0;           	% the discrete time variable
    Tf = 1.0;          	% final simulation time
    ds = 0.0001; 	    % write time of the results

    dx = 2*Lx/(Nx+1);   		% distance between two physical points
    x = (0:Nx)'*dx;             % physical space discretization

    dy = 2*Ly/(Ny+1);   		% distance between two physical points
    y = (0:Ny)'*dy;             % physical space discretization

    [X,Y] = meshgrid(x,y);	% 2D composed grid

    % ------------------------------------------------------------ %

    % vectors of wavenumbers in the transformed space:
    kx = [0:Nx/2 -Nx/2:-1]'*pi/Lx;
    ky = [0:Ny/2 -Ny/2:-1]'*pi/Ly;

    % antialising treatment
    AliasCor = ones(Nx+1, Ny+1);
    jx = (Nx/4+1:Nx/4*3+1);  % the frequencies we sacrify
    jy = (Ny/4+1:Ny/4*3+1);  % the frequencies we sacrify
    AliasCor(jx, :) = 0;
    AliasCor(:, jy) = 0;

    % ------------------------------------------------------------ %

    %%% Some operators arising in NS equations:
    [Kx, Ky] = meshgrid(kx,ky);
    K2 = sparse(Kx.^2 + Ky.^2);     % to compute the Laplace operator

    K2inv = zeros(size(K2));
    K2inv(K2 ~= 0) = 1./K2(K2 ~= 0);

    % ------------------------------------------------------------ %

    fftw('planner', 'hybrid');

    % ------------------------------------------------------------ %
    vx = zeros(Ny+1, Nx+1);
    vy = zeros(Ny+1, Nx+1);
    Vx_hat = fft2(vx);
    Vy_hat = fft2(vy);
    % ------------------------------------------------------------ %

    while (t < Tf-0.5*ds) % main loop in time
%         disp(t);
% %%%%%%%%%%%%%%Implicit Back Euler Method%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         [f1, f2] = force(m, t+ds);  
%         F1_hat = fft2(f1);
%         F2_hat = fft2(f2);    
% 
%         vxx = real(ifft2(1i*Kx.*Vx_hat));
%         vxy = real(ifft2(1i*Ky.*Vx_hat));
% 
%         vyx = real(ifft2(1i*Kx.*Vy_hat));
%         vyy = real(ifft2(1i*Ky.*Vy_hat));
% 
%         Vvx = vx.*vxx + vy.*vxy;
%         Vvy = vx.*vyx + vy.*vyy;
% 
%         Vvx_hat = AliasCor.*fft2(Vvx);
%         Vvy_hat = AliasCor.*fft2(Vvy);
%         
%         Vx_hat = (1/ds*Vx_hat+F1_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Kx-Vvx_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Kx)./(1/ds+nu*K2);
%         Vy_hat = (1/ds*Vy_hat+F2_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Ky-Vvy_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Ky)./(1/ds+nu*K2);
%    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Explicit Euler Method (Unstable with big viscousity nu) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         [f1, f2] = force(m, t);    
%         F1_hat = fft2(f1);
%         F2_hat = fft2(f2);    
%         
%         Vx_hat = fft2(vx);
%         Vy_hat = fft2(vy);
%     
%         vxx = real(ifft2(1i*Kx.*Vx_hat));
%         vxy = real(ifft2(1i*Ky.*Vx_hat));
%     
%         vyx = real(ifft2(1i*Kx.*Vy_hat));
%         vyy = real(ifft2(1i*Ky.*Vy_hat));
%     
%         Vvx = vx.*vxx + vy.*vxy;
%         Vvy = vx.*vyx + vy.*vyy;
%     
%         Vvx_hat = AliasCor.*fft2(Vvx);
%         Vvy_hat = AliasCor.*fft2(Vvy);
%         Vx_hat = ((1/ds-nu*K2).*Vx_hat+F1_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Kx -Vvx_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Kx)*ds;
%         Vy_hat = ((1/ds-nu*K2).*Vy_hat+F2_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Ky -Vvy_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Ky)*ds;
%         
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RK2/CN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         alpha = 1;
%         beta = 0.5;
%         [f1, f2] = force(m, t+alpha*ds);
%         F1_hat = fft2(f1);
%         F2_hat = fft2(f2);    
%         
%         Vx_hat = fft2(vx);
%         Vy_hat = fft2(vy);
%     
%         vxx = real(ifft2(1i*Kx.*Vx_hat));
%         vxy = real(ifft2(1i*Ky.*Vx_hat));
%     
%         vyx = real(ifft2(1i*Kx.*Vy_hat));
%         vyy = real(ifft2(1i*Ky.*Vy_hat));
%     
%         Vvx = vx.*vxx + vy.*vxy;
%         Vvy = vx.*vyx + vy.*vyy;
%     
%         Vvx_hat = AliasCor.*fft2(Vvx);
%         Vvy_hat = AliasCor.*fft2(Vvy);
%         Nunx = F1_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Kx-Vvx_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Kx;
%         Nuny = F2_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Ky-Vvy_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Ky;
%         Vx_hat_1 = (alpha*Nunx+(-beta*nu*K2+1/ds).*Vx_hat)./(1/ds+(alpha-beta)*nu*K2);
%         Vy_hat_1 = (alpha*Nuny+(-beta*nu*K2+1/ds).*Vy_hat)./(1/ds+(alpha-beta)*nu*K2);    
%     
%         [f1, f2] = force(m, t+ds);
%         F1_hat = fft2(f1);
%         F2_hat = fft2(f2);
%         
%         vxx = real(ifft2(1i*Kx.*Vx_hat_1));
%         vxy = real(ifft2(1i*Ky.*Vx_hat_1));
%     
%         vyx = real(ifft2(1i*Kx.*Vy_hat_1));
%         vyy = real(ifft2(1i*Ky.*Vy_hat_1));
%     
%         Vvx = vx.*vxx + vy.*vxy;
%         Vvy = vx.*vyx + vy.*vyy;
%     
%         Vvx_hat = AliasCor.*fft2(Vvx);
%         Vvy_hat = AliasCor.*fft2(Vvy);
%         
%         Nu1x = F1_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Kx-Vvx_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Kx;
%         Nu1y = F2_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Ky-Vvy_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Ky;
%         Vx_hat = ((2*alpha-1)*Nunx+Nu1x+(-alpha*nu*K2+2*alpha/ds).*Vx_hat)./(2*alpha/ds+alpha*nu*K2);
%         Vy_hat = ((2*alpha-1)*Nuny+Nu1y+(-alpha*nu*K2+2*alpha/ds).*Vy_hat)./(2*alpha/ds+alpha*nu*K2); 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% RK3/CN 
% 
%%%%%%%%%%%%%%%%RK3/CN%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         [f1, f2] = force(m, t);
%         F1_hat = fft2(f1);
%         F2_hat = fft2(f2);    
%     
%         ghatx_0 = F1_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Kx;
%         ghaty_0 = F2_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Ky;
%         
%         [f1, f2] = force(m, t+8/15*ds);
%         F1_hat = fft2(f1);
%         F2_hat = fft2(f2);
%         
%         ghatx_1 = F1_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Kx;
%         ghaty_1 = F2_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Ky;
%         
%         vxx = real(ifft2(1i*Kx.*Vx_hat));
%         vxy = real(ifft2(1i*Ky.*Vx_hat));
%     
%         vyx = real(ifft2(1i*Kx.*Vy_hat));
%         vyy = real(ifft2(1i*Ky.*Vy_hat));
%     
%         Vvx = vx.*vxx + vy.*vxy;
%         Vvy = vx.*vyx + vy.*vyy;
%     
%         Vvx_hat = AliasCor.*fft2(Vvx);
%         Vvy_hat = AliasCor.*fft2(Vvy);
%         
%         Chatx_0 = -Vvx_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Kx;
%         Chaty_0 = -Vvy_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Ky;
%         
%         Vx_hat_1 = (8/15*Chatx_0+(-29/96*nu*K2+1/ds).*Vx_hat + 29/96*ghatx_0 + 37/160*ghatx_1)./(1/ds+37/160*nu*K2);
%         Vy_hat_1 = (8/15*Chaty_0+(-29/96*nu*K2+1/ds).*Vy_hat + 29/96*ghaty_0 + 37/160*ghaty_1)./(1/ds+37/160*nu*K2);    
%         
%         [f1, f2] = force(m, t+2/3*ds);   
%         F1_hat = fft2(f1);
%         F2_hat = fft2(f2);    
%     
%         ghatx_0 = ghatx_1;
%         ghaty_0 = ghaty_1;
%         
%         ghatx_1 = F1_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Kx;
%         ghaty_1 = F2_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Ky;
%         
%         vxx = real(ifft2(1i*Kx.*Vx_hat_1));
%         vxy = real(ifft2(1i*Ky.*Vx_hat_1));
%     
%         vyx = real(ifft2(1i*Kx.*Vy_hat_1));
%         vyy = real(ifft2(1i*Ky.*Vy_hat_1));
%     
%         vx = real(ifft2(Vx_hat_1));
%         vy = real(ifft2(Vy_hat_1));
%         
%         Vvx = vx.*vxx + vy.*vxy;
%         Vvy = vx.*vyx + vy.*vyy;
%     
%         Vvx_hat = AliasCor.*fft2(Vvx);
%         Vvy_hat = AliasCor.*fft2(Vvy);
%         
%         Chatx_1 = -Vvx_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Kx;
%         Chaty_1 = -Vvy_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Ky;
%         
%         Vx_hat_2 = (5/12*Chatx_1-17/60*Chatx_0+(3/40*nu*K2+1/ds).*Vx_hat_1-3/40*ghatx_0 + 5/24*ghatx_1)./(1/ds+5/24*nu*K2);
%         Vy_hat_2 = (5/12*Chaty_1-17/60*Chaty_0+(3/40*nu*K2+1/ds).*Vy_hat_1-3/40*ghaty_0 + 5/24*ghaty_1)./(1/ds+5/24*nu*K2); 
%     
%         [f1, f2] = force(m, t+ds);   
%         F1_hat = fft2(f1);
%         F2_hat = fft2(f2);    
%         
%         ghatx_0 = ghatx_1;
%         ghaty_0 = ghaty_1;
%         
%         ghatx_1 = F1_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Kx;
%         ghaty_1 = F2_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Ky;
%         
%         vxx = real(ifft2(1i*Kx.*Vx_hat_2));
%         vxy = real(ifft2(1i*Ky.*Vx_hat_2));
%     
%         vyx = real(ifft2(1i*Kx.*Vy_hat_2));
%         vyy = real(ifft2(1i*Ky.*Vy_hat_2));
%     
%         vx = real(ifft2(Vx_hat_2));
%         vy = real(ifft2(Vy_hat_2));    
%         
%         Vvx = vx.*vxx + vy.*vxy;
%         Vvy = vx.*vyx + vy.*vyy;
%     
%         Vvx_hat = AliasCor.*fft2(Vvx);
%         Vvy_hat = AliasCor.*fft2(Vvy);
%         
%         Chatx_0 = Chatx_1;
%         Chaty_0 = Chaty_1;
%         
%         Chatx_1 = -Vvx_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Kx;
%         Chaty_1 = -Vvy_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Ky;    
%         
%         Vx_hat = (3/4*Chatx_1-5/12*Chatx_0+(-1/6*nu*K2+1/ds).*Vx_hat_2 + 1/6*ghatx_0 + 1/6*ghatx_1)./(1/ds+1/6*nu*K2);
%         Vy_hat = (3/4*Chaty_1-5/12*Chaty_0+(-1/6*nu*K2+1/ds).*Vy_hat_2 + 1/6*ghaty_0 + 1/6*ghaty_1)./(1/ds+1/6*nu*K2); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Fourth-order explicit RK (Not stable), use nu <= 0.001 or time step <
%  0.001
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [Hx, Hy] = rhs(Vx_hat, Vy_hat, m, t);
        Qx = Hx*ds;
        Qy = Hy*ds;
        Vx_hat = Vx_hat + 0.1028639988105*Qx;
        Vy_hat = Vy_hat + 0.1028639988105*Qy;
        
        [Hx, Hy] = rhs(Vx_hat, Vy_hat, m, t+0.1028639988105*ds);
        Qx = -0.4801594388478*Qx + Hx*ds;
        Qy = -0.4801594388478*Qy + Hy*ds;
        Vx_hat = Vx_hat + 0.7408540575767*Qx;
        Vy_hat = Vy_hat + 0.7408540575767*Qy;
        
        [Hx, Hy] = rhs(Vx_hat, Vy_hat, m, t+0.487989987833*ds);
        Qx = -1.4042471952*Qx+Hx*ds;
        Qy = -1.4042471952*Qy+Hy*ds;
        Vx_hat = Vx_hat + 0.7426530946684*Qx;
        Vy_hat = Vy_hat + 0.7426530946684*Qy;

        [Hx, Hy] = rhs(Vx_hat, Vy_hat, m, t+0.6885177231562*ds);
        Qx = -2.016477077503*Qx+Hx*ds;
        Qy = -2.016477077503*Qy+Hy*ds;
        Vx_hat = Vx_hat + 0.4694937902358*Qx;
        Vy_hat = Vy_hat + 0.4694937902358*Qy;
        
        [Hx, Hy] = rhs(Vx_hat, Vy_hat, m, t+0.9023816453077*ds);
        Qx = -1.056444269767*Qx+Hx*ds;
        Qy = -1.056444269767*Qy+Hy*ds;
        Vx_hat = Vx_hat + 0.1881733382888.*Qx;
        Vy_hat = Vy_hat + 0.1881733382888.*Qy;        

    % %%%%%%%%%%%%%%%AB/BDI2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % if t == 0
    %     f1 = force1(m, t+ds);
    %     f2 = force2(m, t+ds);     
    %     F1_hat = fft2(f1);
    %     F2_hat = fft2(f2);    
    % 
    %     Vx_hat_0 = fft2(vx);
    %     Vy_hat_0 = fft2(vy);
    % 
    %     vxx = real(ifft2(1i*Kx.*Vx_hat_0));
    %     vxy = real(ifft2(1i*Ky.*Vx_hat_0));
    % 
    %     vyx = real(ifft2(1i*Kx.*Vy_hat_0));
    %     vyy = real(ifft2(1i*Ky.*Vy_hat_0));
    % 
    %     Vvx = vx.*vxx + vy.*vxy;
    %     Vvy = vx.*vyx + vy.*vyy;
    % 
    %     Vvx_hat = fft2(Vvx);
    %     Vvy_hat = fft2(Vvy);
    %     
    %     Nunx_0 = -Vvx_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Kx;
    %     Nuny_0 = -Vvy_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Ky;
    %     
    %     Vx_hat_1 = (1/ds*Vx_hat_0+F1_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Kx+Nunx_0)./(1/ds+nu*K2);
    %     Vy_hat_1 = (1/ds*Vy_hat_0+F2_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Ky+Nuny_0)./(1/ds+nu*K2);
    %     
    %     vx = real(ifft2(Vx_hat_1));
    %     vy = real(ifft2(Vy_hat_1));
    %     
    %     t = t + ds;
    % end
    % 
    %     f1 = force1(m, t+ds);
    %     f2 = force2(m, t+ds);     
    %     F1_hat = fft2(f1);
    %     F2_hat = fft2(f2);    
    % 
    %     vxx = real(ifft2(1i*Kx.*Vx_hat_1));
    %     vxy = real(ifft2(1i*Ky.*Vx_hat_1));
    % 
    %     vyx = real(ifft2(1i*Kx.*Vy_hat_1));
    %     vyy = real(ifft2(1i*Ky.*Vy_hat_1));
    % 
    %     Vvx = vx.*vxx + vy.*vxy;
    %     Vvy = vx.*vyx + vy.*vyy;
    % 
    %     Vvx_hat = fft2(Vvx);
    %     Vvy_hat = fft2(Vvy);
    % 
    %     Nunx_1 = -Vvx_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Kx;
    %     Nuny_1 = -Vvy_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Ky;
    %     
    %     Vx_hat_2 = (F1_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Kx+4/2/ds*Vx_hat_1-1/2/ds*Vx_hat_0-2*Nunx_1+Nunx_0)./(3/2/ds+nu*K2);
    %     Vy_hat_2 = (F2_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Ky+4/2/ds*Vy_hat_1-1/2/ds*Vy_hat_0-2*Nuny_1+Nuny_0)./(3/2/ds+nu*K2);
    % 
    %     Nunx_0 = Nunx_1;
    %     Nuny_0 = Nuny_1;
    %     
    %     Vx_hat_0 = Vx_hat_1;
    %     Vy_hat_0 = Vy_hat_1;
    %     Vx_hat_1 = Vx_hat_2;
    %     Vy_hat_1 = Vy_hat_2;
    % 
    %     Vx_hat = Vx_hat_1;
    %     Vy_hat = Vy_hat_1;
    %    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         vx = real(ifft2(Vx_hat));
%         vy = real(ifft2(Vy_hat));     
% 
%         Plot(vy, t); % and we plot the solution
        t = t + ds;

%         if abs(t-0.5) < 1e-8
%             QoI = int(Vx_hat, Vy_hat, 1.5);
%         end
    end 

    N=2048;
    Vx_hat = fftshift(Vx_hat);vx
    Vy_hat = fftshift(Vy_hat);
    Vx_hat = padarray(Vx_hat, [(N-Nx)/2, (N-Ny)/2], 0, 'both');
    Vy_hat = padarray(Vy_hat, [(N-Nx)/2, (N-Ny)/2], 0, 'both');
    Vx_hat = ifftshift(Vx_hat*((N+1)/(Nx+1))^2);
    Vy_hat = ifftshift(Vy_hat*((N+1)/(Ny+1))^2);
    vx = real(ifft2(Vx_hat));
    vy = real(ifft2(Vy_hat));
   
    writematrix(vx,'vx');
    writematrix(vy,'vy');    
    
%     vx = real(ifft2(Vx_hat));
%     vy = real(ifft2(Vy_hat));     
% 
%     Plot(vy, t); % and we plot the solution
    %%%%%%%%%%%%%%error compute relative to reference results %%%%%%%%%%%%%
%     Obs = int(Vx_hat, Vy_hat, 0.5);
    return
end



