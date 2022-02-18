
N = 8;
L = 2*pi;

dx = L/(N+1);
x = (0:N)'*dx;
y = x;
[X, Y] = meshgrid(x, y);

z = cos(X).*sin(Y);
test1 = fft2(z);
zk = fftshift(fft2(z));

N2 = 1024;
dx = L/(N2+1);
x2 = (0:N2)'*dx;
y2 = x2;
[X2, Y2] = meshgrid(x2, y2);
zk2 = padarray(zk, [(N2-N)/2, (N2-N)/2], 0, 'both');
test2 = ifftshift(zk2);
z2 = ifft2(ifftshift(((N2+1)/(N+1))^2*zk2));
figure(1)
contourf(X, Y, z, 200, 'LineColor', 'none');
colorbar;

figure(2)
contourf(X2, Y2, z2, 200, 'LineColor', 'none');
colorbar;

