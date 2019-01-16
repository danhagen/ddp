%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  Iterative Linear Quadratic Regulator for Cart-Pole Rigid Body Dynamics          %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  Course: DDP Derivation and Control System Studies                               %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  Author: Evangelos Theodorou                                                     %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  Modified by: Manan Gandhi                                                       %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

global m;  %Changed the initial paramaters to reflect the new problem.
global M;
global l;
global gr;
global h

%h is the stp used to determine the derivative
h = .000001;

% Cart Pole Parameters
% masses in Kgr
m = 1;
M = 10;

%gravity m/sec^2
gr = 9.81;

% length parameters in meters
l = 1.5;

% Horizon
Horizon = 300; % 1.5sec
% Number of Iterations
num_iter = 100;

% Discretization
dt = 0.01;
global timeee
timeee = Horizon*dt;

% Weight in Final State:
Q_f = zeros(4,4);
Q_f(1,1) = 5;
Q_f(2,2) = 1000;
Q_f(3,3) = 5;
Q_f(4,4) = 50;

% Weight in the Control:
R = 1e-3; %Modified form original because our control is only one dimensional.

% Initial Configuration:
xo = zeros(4,1);
xo(1,1) = 0;
xo(2,1) = pi;

% Initial Control:
u_k = zeros(1,Horizon-1);
du_k = zeros(1,Horizon-1);


% Initial trajectory:
x_traj = zeros(4,Horizon);


% Target:
p_target(1,1) = 0;
p_target(2,1) = 0;
p_target(3,1) = 0;
p_target(4,1) = 0;


% Learning Rate:
gamma = 0.2;


for k = 1:num_iter
    
    %------------------------------------------------> Linearization of the dynamics
    %------------------------------------------------> Quadratic Approximations of the cost function
    for  j = 1:(Horizon-1)
        
        [l0,l_x,l_xx,l_u,l_uu,l_ux] = fnCost(x_traj(:,j), u_k(:,j), j,R,dt);
        q0(j) = dt * l0;
        q_k(:,j) = dt * l_x;
        Q_k(:,:,j) = dt * l_xx;
        r_k(:,j) = dt * l_u;
        R_k(:,:,j) = dt * l_uu;
        P_k(:,:,j) = dt * l_ux;
        
        [dfx,dfu] = fnState_And_Control_Transition_Matrices(x_traj(:,j),u_k(:,j),du_k(:,j),dt);
        
        A(:,:,j) = eye(4,4) + dfx * dt;
        B(:,:,j) = dfu * dt;
    end
    
    %------------------------------------------------> Find the controls
    Vxx(:,:,Horizon)= Q_f;
    Vx(:,Horizon) = Q_f * (x_traj(:,Horizon) - p_target);
    V(Horizon) = (x_traj(:,Horizon) - p_target)' * Q_f * (x_traj(:,Horizon) - p_target);
    
    
    %------------------------------------------------> Backpropagation of the Value Function
    for j = (Horizon-1):-1:1
        
        H = R_k(:,:,j) + B(:,:,j)' * Vxx(:,:,j+1) * B(:,:,j);
        G = P_k(:,:,j) + B(:,:,j)' * Vxx(:,:,j+1) * A(:,:,j);
        g = r_k(:,j) +  B(:,:,j)' * Vx(:,j+1);
        
        
        inv_H = inv(H);
        L_k(:,:,j)= - inv_H * G;
        l_k (:,j) = - inv_H *g;
        
        
        Vxx(:,:,j) = Q_k(:,:,j)+ A(:,:,j)' * Vxx(:,:,j+1) * A(:,:,j) + L_k(:,:,j)' * H * L_k(:,:,j) + L_k(:,:,j)' * G + G' * L_k(:,:,j);
        Vx(:,j)= q_k(:,j) +  A(:,:,j)' *  Vx(:,j+1) + L_k(:,:,j)' * g + G' * l_k(:,j) + L_k(:,:,j)'*H * l_k(:,j);
        V(:,j) = q0(j) + V(j+1)   +   0.5 *  l_k (:,j)' * H * l_k (:,j) + l_k (:,j)' * g;
    end
    
    
    %----------------------------------------------> Find the controls
    dx = zeros(4,1);
    for i=1:(Horizon-1)
        du = l_k(:,i) + L_k(:,:,i) * dx;
        dx = A(:,:,i) * dx + B(:,:,i) * du;
        u_new(:,i) = u_k(:,i) + gamma * du;
    end
    
    u_k = u_new;
    
    
    %---------------------------------------------> Simulation of the Nonlinear System
    [x_traj] = fnsimulate(xo,u_new,Horizon,dt);
    [Cost(:,k)] =  fnCostComputation(x_traj,u_k,p_target,dt,Q_f,R);
    x1(k,:) = x_traj(1,:);
    
    
    fprintf('iLQG Iteration %d,  Current Cost = %e \n',k,Cost(1,k));
    
    
end



time(1)=0;
for i= 2:Horizon
    time(i) =time(i-1) + dt;
end


%---------------------------------------------> Plot Section
figure(1);
subplot(3,2,1)
hold on
plot(time,x_traj(1,:),'linewidth',4);
plot(time,p_target(1,1)*ones(1,Horizon),'red','linewidth',4)
title('X Position','fontsize',20);
xlabel('Time in sec','fontsize',20)
hold off;
grid;


subplot(3,2,2);hold on;
plot(time,x_traj(2,:),'linewidth',4);
plot(time,p_target(2,1)*ones(1,Horizon),'red','linewidth',4)
title('Theta','fontsize',20);
xlabel('Time in sec','fontsize',20)
hold off;
grid;





subplot(3,2,3);hold on
plot(time,x_traj(3,:),'linewidth',4);
plot(time,p_target(3,1)*ones(1,Horizon),'red','linewidth',4)
title('X velocity','fontsize',20)
xlabel('Time in sec','fontsize',20)
hold off;
grid;

subplot(3,2,4);hold on
plot(time,x_traj(4,:),'linewidth',4);
plot(time,p_target(4,1)*ones(1,Horizon),'red','linewidth',4)
title('Angular Velocity','fontsize',20)
xlabel('Time in sec','fontsize',20)
hold off;
grid;

subplot(3,2,5);hold on
plot(Cost,'linewidth',2);
xlabel('Iterations','fontsize',20)
title('Cost','fontsize',20);

run('animate.m')
max(x_traj(1,:))
min(x_traj(1,:))
max(x_traj(2,:))
min(x_traj(2,:))
max(x_traj(3,:))
min(x_traj(3,:))
max(x_traj(4,:))
min(x_traj(4,:))
max(u_k)
min(u_k)
