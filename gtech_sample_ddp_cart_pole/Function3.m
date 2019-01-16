function dx2 = Function3(x1,x2,x3,x4,F)
global M m l
% pos = x1;
% theta= x2;
% vel = x3;
% ang_vel = x4;
% num2 = m*l*(ang_vel^2*sin(theta)-Function4(pos,theta,vel,ang_vel,F)*cos(theta));
% den = (M+m);
% dx2 = (F+num2)/den;

dx2 = (F+m*l*(x4^2*sin(x2)-Function4(x1,x2,x3,x4,F)*cos(x2)))/(m+M);
end