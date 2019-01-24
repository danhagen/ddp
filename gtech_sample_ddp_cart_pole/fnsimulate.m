function [x] = fnsimulate(xo,u_new,Horizon,dt)

global m;  %Changed the initial paramaters to reflect the new problem.
global M;
global l
global gr;


x = xo;

for k = 1:(Horizon-1)


      Fx(1,1) = x(3,k);
      Fx(2,1) = x(4,k);
%       Fx(3,1) = ((l*x(4,k)^2*sin(x(2,k))-((gr*sin(x(2,k))-cos(x(2,k))*(u_new(k)+m*l*x(4,k)^2*sin(x(2,k))))/(l*(1-(m*cos(x(2,k))^2)/(M+m))))*l*cos(x(2,k)))*m/(m+M))+u_new(k)/(m+M);
%       Fx(4,1) = (gr*sin(x(2,k))-cos(x(2,k))*(u_new(k)+m*l*x(4,k)^2*sin(x(2,k))))/(l*(1-(m*cos(x(2,k))^2)/(M+m)));
      Fx(3,1) = Function3(x(1,k),x(2,k),x(3,k),x(4,k),u_new(k));
      Fx(4,1) = Function4(x(1,k),x(2,k),x(3,k),x(4,k),u_new(k));

x(:,k+1) = x(:,k) + Fx * dt;
end
