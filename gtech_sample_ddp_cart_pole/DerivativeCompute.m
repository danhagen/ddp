clear
clc
global m M l g
syms x1 x2 x3 x4 dx1 dx2 dx3 dx4 m M l g F A B;
h = .00001;
A = vpa(zeros(4,4));

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
A(3,1) = (Function3(x1,x2,x3,x4,F)-Function3(x1-h,x2,x3,x4,F))/h;
A(3,2) = (Function3(x1,x2,x3,x4,F)-Function3(x1,x2-h,x3,x4,F))/h;
A(3,3) = (Function3(x1,x2,x3,x4,F)-Function3(x1,x2,x3-h,x4,F))/h;
A(3,4) = (Function3(x1,x2,x3,x4,F)-Function3(x1,x2,x3,x4-h,F))/h;

%F4 is the angular acceleration of the pendulum.
A(4,1) = (Function4(x1,x2,x3,x4,F)-Function4(x1-h,x2,x3,x4,F))/h;
A(4,2) = (Function4(x1,x2,x3,x4,F)-Function4(x1,x2-h,x3,x4,F))/h;
A(4,3) = (Function4(x1,x2,x3,x4,F)-Function4(x1,x2,x3-h,x4,F))/h;
A(4,4) = (Function4(x1,x2,x3,x4,F)-Function4(x1,x2,x3,x4-h,F))/h;

%Build the B matrix
B = vpa(zeros(4,1));
B(1,1) = 0;
B(2,1) = 0;
B(3,1) = (Function3(x1,x2,x3,x4,F)-Function3(x1,x2,x3,x4,F-h))/h;
B(4,1) = (Function4(x1,x2,x3,x4,F)-Function4(x1,x2,x3,x4,F-h))/h;
