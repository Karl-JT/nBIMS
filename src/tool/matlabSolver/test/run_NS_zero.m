format long
%function [QoIfinal, Obs] = run_NS(m)

global dy nu Kx Ky K2 Lx Ly Nx Ny Nxy X Y

% m = [1.0, 0.5, 0.3, 0.8];
m = [1.0];

nu = 1;

Lx = 0.5;
Ly = 0.5;  

%%% Numerical parameters:
Nx = 128;  % number of Fourier modes in discrete solution x-dir
Ny = 128;	% number of Fourier modes in discrete solution y-dir
Nxy = (Nx+1)*(Ny+1);    

alpha = 0.5;
beta = 0.25;
t = 0.0;           	% the discrete time variable
Tf = 1.0;          	% final simulation time
ds = 0.00625; 			% write time of the results

dx = 2*Lx/(Nx+1);   		% distance between two physical points
x = (0:Nx)'*dx;  % physical space discretization

dy = 2*Ly/(Ny+1);   		% distance between two physical points
y = (0:Ny)'*dy;  % physical space discretization

[X,Y] = meshgrid(x,y);	% 2D composed grid

% ------------------------------------------------------------ %

% vectors of wavenumbers in the transformed space:
kx = [0:Nx/2 -Nx/2:-1]'*pi/Lx;
ky = [0:Ny/2 -Ny/2:-1]'*pi/Ly;

% antialising treatment
jx = (Nx/4+1:Nx/4*3+1);  % the frequencies we sacrify
kx(jx) = 0;

jy = (Ny/4+1:Ny/4*3+1);  % the frequencies we sacrify
ky(jy) = 0;

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

% ------------------------------------------------------------ %

while (t < Tf+0.5*ds) % main loop in time
    disp(t);

    f1 = force1(m, t+alpha*ds);
    f2 = force2(m, t+alpha*ds);     
    F1_hat = fft2(f1);
    F2_hat = fft2(f2);    
    
    Vx_hat = fft2(vx);
    Vy_hat = fft2(vy);

    vxx = real(ifft2(1i*Kx.*Vx_hat));
    vxy = real(ifft2(1i*Ky.*Vx_hat));

    vyx = real(ifft2(1i*Kx.*Vy_hat));
    vyy = real(ifft2(1i*Ky.*Vy_hat));

    Vvx = vx.*vxx + vy.*vxy;
    Vvy = vx.*vyx + vy.*vyy;

    Vvx_hat = fft2(Vvx);
    Vvy_hat = fft2(Vvy);

%%%%%%%%%%%%%%%Back Euler Method%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Vx_hat = (1/ds*Vx_hat+F1_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Kx-Vvx_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Kx)./(1/ds+nu*K2);
%     Vy_hat = (1/ds*Vy_hat+F2_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Ky-Vvy_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Ky)./(1/ds+nu*K2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%RK2/CN%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Nunx = F1_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Kx-Vvx_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Kx;
    Nuny = F2_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Ky-Vvy_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Ky;
    Vx_hat_1 = (alpha*Nunx+(-beta*nu*K2+1/ds).*Vx_hat)./(1/ds+(alpha-beta)*nu*K2);
    Vy_hat_1 = (alpha*Nuny+(-beta*nu*K2+1/ds).*Vy_hat)./(1/ds+(alpha-beta)*nu*K2);    

    f1 = force1(m, t+ds);
    f2 = force2(m, t+ds);     
    F1_hat = fft2(f1);
    F2_hat = fft2(f2);
    
    vxx = real(ifft2(1i*Kx.*Vx_hat_1));
    vxy = real(ifft2(1i*Ky.*Vx_hat_1));

    vyx = real(ifft2(1i*Kx.*Vy_hat_1));
    vyy = real(ifft2(1i*Ky.*Vy_hat_1));

    Vvx = vx.*vxx + vy.*vxy;
    Vvy = vx.*vyx + vy.*vyy;

    Vvx_hat = fft2(Vvx);
    Vvy_hat = fft2(Vvy);
    
    Nu1x = F1_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Kx-Vvx_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Kx;
    Nu1y = F2_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Ky-Vvy_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Ky;
    Vx_hat = ((2*alpha-1)*Nunx+Nu1x+(-alpha*nu*K2+2*alpha/ds).*Vx_hat)./(2*alpha/ds+alpha*nu*K2);
    Vy_hat = ((2*alpha-1)*Nuny+Nu1y+(-alpha*nu*K2+2*alpha/ds).*Vy_hat)./(2*alpha/ds+alpha*nu*K2); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    vx = real(ifft2(Vx_hat));
    vy = real(ifft2(Vy_hat));

    Plot(vx, t); % and we plot the solution
    
    t = t + ds;
end 

%%%%%%%%%%%%%%error compute relative to reference results %%%%%%%%%%%%%
NC = 6;
Nn = 399; %Nn has to be odd number
N = Nn*(NC+1) - 1;
dL = Lx*2/(N+1);
x = (0:N)'*dL;
y = x;
[X, Y] = meshgrid(x, y);

% utest = -m(1)*cos(2*pi*X).*sin(2*pi*Y)*(exp(1)-1);
% vtest = m(1)*sin(2*pi*X).*cos(2*pi*Y)*(exp(1)-1);
% 
% Vx_hat = fftshift(Vx_hat);
% Vy_hat = fftshift(Vy_hat);
% Vx_hat = padarray(Vx_hat, [(N-Nx)/2, (N-Ny)/2], 0, 'both');
% Vy_hat = padarray(Vy_hat, [(N-Nx)/2, (N-Ny)/2], 0, 'both');
% vx = real(ifft2(ifftshift(Vx_hat*((N+1)/(Nx+1))^2)));
% vy = real(ifft2(ifftshift(Vy_hat*((N+1)/(Ny+1))^2)));
% 
% uerror = abs(vx-utest);
% verror = abs(vy-vtest);
% 
% figure(2)
% Plot(vx, t-ds); % and we plot the solution
% 
% uerror = [uerror, uerror(:, 1)];
% uerror = [uerror; uerror(1, :)];
% uerror(end, end) = uerror(1, 1);
% 
% verror = [verror, verror(:, 1)];
% verror = [verror; verror(1, :)];
% verror(end, end) = verror(1, 1);
% 
% terror = uerror + verror;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = [x; 1];
y = x;
[X, Y] = meshgrid(x, y);

vxx_hat = 1i*Kx.*Vx_hat;
vxy_hat = 1i*Ky.*Vx_hat;

vxx_hat = fftshift(vxx_hat);
vxy_hat = fftshift(vxy_hat);

vxx_hat = padarray(vxx_hat, [(N-Nx)/2, (N-Ny)/2], 0, 'both');
vxy_hat = padarray(vxy_hat, [(N-Nx)/2, (N-Ny)/2], 0, 'both');
vxx_hat = ifftshift(vxx_hat*((N+1)/(Nx+1))^2);
vxy_hat = ifftshift(vxy_hat*((N+1)/(Ny+1))^2);
vxx = real(ifft2(vxx_hat));
vxy = real(ifft2(vxy_hat));

vxx = [vxx, vxx(:, 1)];
vxx = [vxx; vxx(1, :)];
vxx(end, end) = vxx(1, 1);
vxy = [vxy, vxy(:, 1)];
vxy = [vxy; vxy(1, :)];
vxy(end, end) = vxy(1, 1);

vyx_hat = 1i*Kx.*Vy_hat;
vyy_hat = 1i*Ky.*Vy_hat;

vyx_hat = fftshift(vyx_hat);
vyy_hat = fftshift(vyy_hat);

vyx_hat = padarray(vyx_hat, [(N-Nx)/2, (N-Ny)/2], 0, 'both');
vyy_hat = padarray(vyy_hat, [(N-Nx)/2, (N-Ny)/2], 0, 'both');
vyx_hat = ifftshift(vyx_hat*((N+1)/(Nx+1))^2);
vyy_hat = ifftshift(vyy_hat*((N+1)/(Ny+1))^2);
vyx = real(ifft2(vyx_hat));
vyy = real(ifft2(vyy_hat));

vyx = [vyx, vyx(:, 1)];
vyx = [vyx; vyx(1, :)];
vyx(end, end) = vyx(1, 1);
vyy = [vyy, vyy(:, 1)];
vyy = [vyy; vyy(1, :)];
vyy(end, end) = vyy(1, 1);

vxIn = X.^0.5.*Y.^0.5.*vxy;
vyIn = X.^0.5.*Y.^0.5.*vyx;

vIn = vxIn - vyIn;

Plot(vIn, t-ds); % and we plot the solution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w = CloseNewtonCotes(NC+1);
w = w*dL;
wtmp = zeros(N+2, 1);
for i = 0:(ceil((N+2)/(NC+1))-2)
    wtmp(i*(NC+1)+1:(i+1)*(NC+1)+1) = wtmp(i*(NC+1)+1:(i+1)*(NC+1)+1)+w;
end
w = wtmp;
w2d = kron(w, w');
QoI = vIn.*w2d;
reg(log2(0.2/ds)) = 100*sum(sum(QoI))


% Plot(vIn, t-ds); % and we plot the solution
 %   return
%end
% 
function f = force1 (m, t)

    global X Y

    f = -m*cos(2*pi*X).*sin(2*pi*Y)*exp(t)-2*m*(2*pi)^2*cos(2*pi*X).*sin(2*pi*Y)*(exp(t)-1)-pi*m^2*sin(4*pi*X)*(exp(t)-1)^2;
    
    return
end

function f = force2 (m, t)

    global X Y

    f = m*sin(2*pi*X).*cos(2*pi*Y)*exp(t)+2*m*(2*pi)^2*sin(2*pi*X).*cos(2*pi*Y)*(exp(t)-1)-pi*m^2*sin(4*pi*Y)*(exp(t)-1)^2;       
    
    return
end


% function f = force1 (m, t)
% 
%     global X Y
% 
%     f = -m(1)*cos(2*pi*X).*sin(2*pi*Y)*exp(t)-m(2)*sin(2*pi*X).*cos(2*pi*Y)*exp(t)-m(3)/2*cos(4*pi*X).*sin(4*pi*Y)*exp(t)-m(4)/2*sin(4*pi*X).*cos(4*pi*Y)*exp(t);
%     
%     return
% end
% 
% function f = force2 (m, t)
% 
%     global X Y
% 
%     f = m(1)*sin(2*pi*X).*cos(2*pi*Y)*exp(t)+m(2)*cos(2*pi*X).*sin(2*pi*Y)*exp(t)+m(3)/2*sin(4*pi*X).*cos(4*pi*Y)*exp(t)+m(4)/2*cos(4*pi*X).*sin(4*pi*Y)*exp(t);       
%     
%     return
% end


function Plot (Om, t)

	global Lx Ly X Y
    
	surf(X, Y, Om), grid off
	shading interp;
	colormap(jet); cc = colorbar;
    xlim([0 2*Lx]); ylim([0 2*Ly]);
    xlabel('$x$', 'interpreter', 'latex', 'fontsize', 12);
    ylabel('$y$', 'interpreter', 'latex', 'fontsize', 12, 'Rotation', 1);
    xlabel(cc, '$\omega(x,y,t)$', 'interpreter', 'latex', 'fontsize', 12, 'Rotation', 90);
    view([0 90]);

    title (['velocity distribution at t = ',num2str(t,'%4.2f')], 'interpreter', 'latex', 'fontsize', 12);

    set(gcf, 'Color', 'w');
    drawnow
end % Plot ()


