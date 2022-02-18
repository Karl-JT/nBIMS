format long
%function [QoIfinal, Obs] = run_NS(m)

global dy nu Kx Ky K2 Lx Ly Nx Ny Nxy X Y

% m = [1.0, 0.5, 0.3, 0.8];
m = 1.0;

nu = 1;

Lx = 0.5;
Ly = 0.5;  

%%% Numerical parameters:
Nx = 192;  % number of Fourier modes in discrete solution x-dir
Ny = 192;	% number of Fourier modes in discrete solution y-dir
Nxy = Nx*Ny;    

t = 0.0;           	% the discrete time variable
Tf = 1.0;          	% final simulation time
ds = 0.1; 			% write time of the results

dx = 2*Lx/Nx;   		% distance between two physical points
x = (1:Nx)'*dx;  % physical space discretization

dy = 2*Ly/Ny;   		% distance between two physical points
y = (1:Ny)'*dy;  % physical space discretization

[X,Y] = meshgrid(x,y);	% 2D composed grid

% ------------------------------------------------------------ %

% vectors of wavenumbers in the transformed space:
kx = [0:Nx/2 1-Nx/2:-1]'*pi/Lx;
ky = [0:Ny/2 1-Ny/2:-1]'*pi/Ly;

% antialising treatment
jx = (Nx/4+2:Nx/4*3);  % the frequencies we sacrify
kx(jx) = 0;

jy = (Ny/4+2:Ny/4*3);  % the frequencies we sacrify
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
vx = zeros(Ny, Nx);
vy = zeros(Ny, Nx);

% ------------------------------------------------------------ %

while (t < Tf+0.5*ds) % main loop in time
    disp(t);

    f1 = force1(m, t);
    f2 = force2(m, t);     
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

    Vx_hat = (1/ds*Vx_hat+F1_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Kx-Vvx_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Kx)./(1/ds+nu*K2);
    Vy_hat = (1/ds*Vy_hat+F2_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Ky-Vvy_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Ky)./(1/ds+nu*K2);

    vx = real(ifft2(Vx_hat));
    vy = real(ifft2(Vy_hat));

    Plot(vx, t); % and we plot the solution
    
    t = t + ds;
end 

%%%%%%%%%%%%%%error compute relative to reference results %%%%%%%%%%%%%
% vx = -m(1)*cos(2*pi*X).*sin(2*pi*Y)*(exp(1)-1);
% vy = m(1)*sin(2*pi*X).*cos(2*pi*Y)*(exp(1)-1);
% 
% Vx_hat = fft2(vx);
% Vy_hat = fft2(vy);

% 
% uerror = abs(vx-utest);
% verror = abs(vy-vtest);
% 
% figure(2)
% Plot(uerror, t); % and we plot the solution


% uerror = [uerror(:, end), uerror];
% uerror = [uerror(end, :); uerror];
% uerror(1, 1) = uerror(end, end);
% 
% verror = [verror(:, end), verror];
% verror = [verror(end, :); verror];
% verror(1, 1) = verror(end, end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vxx = real(ifft2(1i*Kx.*Vx_hat));
vxy = real(ifft2(1i*Ky.*Vx_hat));
vxx = [vxx(:, end), vxx];
vxx = [vxx(end, :); vxx];
vxx(1, 1) = vxx(end, end);
vxy = [vxy(:, end), vxy];
vxy = [vxy(end, :); vxy];
vxy(1, 1) = vxy(end, end);

vyx = real(ifft2(1i*Kx.*Vy_hat));
vyy = real(ifft2(1i*Ky.*Vy_hat));
vyx = [vyx(:, end), vyx];
vyx = [vyx(end, :); vyx];
vyx(1, 1) = vyx(end, end);
vyy = [vyy(:, end), vyy];
vyy = [vyy(end, :); vyy];
vyy(1, 1) = vyy(end, end);


vx = [vx(:, end), vx];
vx = [vx(end, :); vx];
vx(1, 1) = vx(end, end);
vy = [vy(:, end), vy];
vy = [vy(end, :); vy];
vy(1, 1) = vy(end, end);


X = [X(:, end)*0, X];
X = [X(end, :)*0; X];
X(1, 1) = 0;
Y = [Y(:, end)*0, Y];
Y = [Y(end, :)*0; Y];
Y(1, 1) = 0;

vxIn = X.^0.5.*Y.^0.5.*vx;
vyIn = X.^0.5.*Y.^0.5.*vy;

vIn = vxIn - vyIn;

Plot(vIn, t-ds); % and we plot the solution

NC = 10;
w1 = CloseNewtonCotes(10);
w2 = CloseNewtonCotes(10);

h = 1/Nx;
w = kron(w1, w2')*h^2;

for i = 1:log2(Nx/NC)
    wtmp = zeros(NC*2^i+1, NC*2^i+1);
    wtmp(1:(end+1)/2, 1:(end+1)/2) = wtmp(1:(end+1)/2, 1:(end+1)/2) + w;
    wtmp((end+1)/2:end, 1:(end+1)/2)= wtmp((end+1)/2:end, 1:(end+1)/2) + w;
    wtmp(1:(end+1)/2, (end+1)/2:end)= wtmp(1:(end+1)/2, (end+1)/2:end) +w;
    wtmp((end+1)/2:end, (end+1)/2:end)= wtmp((end+1)/2:end, (end+1)/2:end) +w;
    w = wtmp;
end

QoI = sum(sum(w.*vIn));
disp(QoI*100);


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


