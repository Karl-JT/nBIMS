function [Hx, Hy] = rhs(Vx_hat, Vy_hat, m, t)

    global Kx Ky nu K2 K2inv AliasCor

    [f1, f2] = force(m, t);
    F1_hat = fft2(f1);
    F2_hat = fft2(f2);    

    vx = real(ifft2(Vx_hat));
    vy = real(ifft2(Vy_hat));
    
    vxx = real(ifft2(1i*Kx.*Vx_hat));
    vxy = real(ifft2(1i*Ky.*Vx_hat));

    vyx = real(ifft2(1i*Kx.*Vy_hat));
    vyy = real(ifft2(1i*Ky.*Vy_hat));

    Vvx = vx.*vxx + vy.*vxy;
    Vvy = vx.*vyx + vy.*vyy;

    Vvx_hat = AliasCor.*fft2(Vvx);
    Vvy_hat = AliasCor.*fft2(Vvy);
   
    Hx = F1_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Kx -Vvx_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Kx-nu*K2.*Vx_hat;
    Hy = F2_hat-(Kx.*F1_hat+Ky.*F2_hat).*K2inv.*Ky -Vvy_hat+(Kx.*Vvx_hat+Ky.*Vvy_hat).*K2inv.*Ky-nu*K2.*Vy_hat;
    
    return
end