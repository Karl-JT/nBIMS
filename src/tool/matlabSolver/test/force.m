% function [f1, f2] = force (m, t)
% 
%     global X Y
% 
%     f1 = -m*cos(2*pi*X).*sin(2*pi*Y)*exp(t)-2*m*(2*pi)^2*cos(2*pi*X).*sin(2*pi*Y)*(exp(t)-1)-pi*m^2*sin(4*pi*X)*(exp(t)-1)^2;
%     f2 = m*sin(2*pi*X).*cos(2*pi*Y)*exp(t)+2*m*(2*pi)^2*sin(2*pi*X).*cos(2*pi*Y)*(exp(t)-1)-pi*m^2*sin(4*pi*Y)*(exp(t)-1)^2;       
% 
%     return
% end

% function [f1, f2] = force (m, t)
% 
%     global X Y nu
% 
%     f1 = -m*cos(2*pi*X).*sin(2*pi*Y)*exp(t)-nu*2*m*(2*pi)^2*cos(2*pi*X).*sin(2*pi*Y)*(exp(t)-1)-pi*m^2*sin(4*pi*X)*(exp(t)-1)^2;
%     f2 = m*sin(2*pi*X).*cos(2*pi*Y)*exp(t)+nu*2*m*(2*pi)^2*sin(2*pi*X).*cos(2*pi*Y)*(exp(t)-1)-pi*m^2*sin(4*pi*Y)*(exp(t)-1)^2;       
% 
%     return
% end

function [f1, f2] = force (m, t)

    global X Y

    f1 = -m(1)*cos(2*pi*X).*sin(2*pi*Y)*exp(t);%-m(2)*sin(2*pi*X).*cos(2*pi*Y)*exp(t)-m(3)/2*cos(4*pi*X).*sin(4*pi*Y)*exp(t)-m(4)/2*sin(4*pi*X).*cos(4*pi*Y)*exp(t);
    f2 = m(1)*sin(2*pi*X).*cos(2*pi*Y)*exp(t);%+m(2)*cos(2*pi*X).*sin(2*pi*Y)*exp(t)+m(3)/2*sin(4*pi*X).*cos(4*pi*Y)*exp(t)+m(4)/2*cos(4*pi*X).*sin(4*pi*Y)*exp(t);       
  
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