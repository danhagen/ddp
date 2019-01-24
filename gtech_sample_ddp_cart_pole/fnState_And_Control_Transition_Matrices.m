function [A,B] = fnState_And_Control_Transition_Matrices(x,u,du,dt)


x1 = x(1,1);
x2 = x(2,1);
x3 = x(3,1);
x4 = x(4,1);

%Removed the u split into two scalars because u is already a scalar.
h = .00000001;
A = zeros(4,4);

%Build the A matrix
%A(1,1) = 0 dF1/x1 dx1 = x3
%A(1,2) = 0 dF1/x2 dx1 = x3
A(1,3) = 1; %dF1/x3 dx1 = x3
%A(1,4) = 0 dF1/x4 dx1 = x3

%A(2,1) = 0 dF2/x1 dx2 = x4
%A(2,2) = 0 dF2/x2 dx2 = x4
%A(2,3) = 0 dF2/x3 dx2 = x4
A(2,4) = 1; %dF2/x4 dx2 = x4

%F3 is the acceleration of the cart.
A(3,1) = (Function3(x1,x2,x3,x4,u)-Function3(x1-h,x2,x3,x4,u))/h;
A(3,2) = (Function3(x1,x2,x3,x4,u)-Function3(x1,x2-h,x3,x4,u))/h;
A(3,3) = (Function3(x1,x2,x3,x4,u)-Function3(x1,x2,x3-h,x4,u))/h;
A(3,4) = (Function3(x1,x2,x3,x4,u)-Function3(x1,x2,x3,x4-h,u))/h;

%F4 is the angular acceleration of the pendulum.
A(4,1) = (Function4(x1,x2,x3,x4,u)-Function4(x1-h,x2,x3,x4,u))/h;
A(4,2) = (Function4(x1,x2,x3,x4,u)-Function4(x1,x2-h,x3,x4,u))/h;
A(4,3) = (Function4(x1,x2,x3,x4,u)-Function4(x1,x2,x3-h,x4,u))/h;
A(4,4) = (Function4(x1,x2,x3,x4,u)-Function4(x1,x2,x3,x4-h,u))/h;

%Build the B matrix
B = zeros(4,1);
B(1,1) = 0;
B(2,1) = 0;
B(3,1) = (Function3(x1,x2,x3,x4,u)-Function3(x1,x2,x3,x4,u-h))/h;
B(4,1) = (Function4(x1,x2,x3,x4,u)-Function4(x1,x2,x3,x4,u-h))/h;
