example = matfile('vxx.mat');
vxx = example.vxx;
% example = matfile('vxy.mat');
% vxy = example.vxy;
% example = matfile('vyx.mat');
% vyx = example.vyx;
% example = matfile('vyy.mat');
% vyy = example.vyy;
% 
% NC = 6;
% Nn = 9; %Nn has to be odd number
% N = Nn*(NC+1) - 1;
N = 4095;
dL = 1/(N+1);
x = (0:N)'*dL;
x = [x; 1];
y = x;
[X, Y] = meshgrid(x, y);

% contourf(X,Y,vxx);

uxx=readmatrix("vx");
% uxx=uxx(:,1:end-1);
% uxy=readmatrix("uxy");
% uxy=uxx(:,1:end-1);
% uyx=readmatrix("uyx");
% uyx=uxx(:,1:end-1);
% uyy=readmatrix("uyy");
% uyy=uxx(:,1:end-1);

contourf(X, Y, uxx);
colorbar;

% intgrand=(uxx-vxx).^2 + (uxy-vxy).^2 + (uyx-vyx).^2 + (uyy-vyy).^2;

% w = CloseNewtonCotes(NC+1);
% w = w*dL;
% wtmp = zeros(N+2, 1);
% for i = 0:(ceil((N+2)/(NC+1))-2)
%     wtmp(i*(NC+1)+1:(i+1)*(NC+1)+1) = wtmp(i*(NC+1)+1:(i+1)*(NC+1)+1)+w;
% end
% w = wtmp;
% w2d = kron(w, w');
% output = sum(sum(intgrand.*w2d));