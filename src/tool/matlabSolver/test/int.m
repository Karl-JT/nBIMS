function output = int(Vx_hat, Vy_hat, power)
    global Lx Kx Ky Nx Ny

    NC = 6;
    Nn = 65;%999; %Nn has to be odd number
    N = Nn*(NC+1) - 1;
    dL = Lx*2/(N+1);
    x = (0:N)'*dL;
    % utest = -m(1)*cos(2*pi*X).*sin(2*pi*Y)*(exp(0.1)-1);
    % vtest = m(1)*sin(2*pi*X).*cos(2*pi*Y)*(exp(0.1)-1);
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
    % % figure(2)
    % % Plot(vx, t-ds); % and we plot the solution
    % 
    % uerror = [uerror, uerror(:, 1)];
    % uerror = [uerror; uerror(1, :)];
    % uerror(end, end) = uerror(1, 1);
    % 
    % verror = [verror, verror(:, 1)];
    % verror = [verror; verror(1, :)];
    % verror(end, end) = verror(1, 1);
    % 
    % terror = uerror+verror;
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

    save("vxx.mat", 'vxx');
    save("vxy.mat", 'vxy');
    save("vyx.mat", 'vyx');
    save("vyy.mat", 'vyy');
    
    vxIn = X.^power.*Y.^power.*vxy;
    vyIn = X.^power.*Y.^power.*vyx;
    
    vIn = vxIn - vyIn;
    % 
    % Plot(vIn, t-ds); % and we plot the solution
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    w = CloseNewtonCotes(NC+1);
    w = w*dL;
    wtmp = zeros(N+2, 1);
    for i = 0:(ceil((N+2)/(NC+1))-2)
        wtmp(i*(NC+1)+1:(i+1)*(NC+1)+1) = wtmp(i*(NC+1)+1:(i+1)*(NC+1)+1)+w;
    end
    w = wtmp;
    w2d = kron(w, w');
    output = 100*sum(sum(vIn.*w2d));
    return
end
