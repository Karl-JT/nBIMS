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