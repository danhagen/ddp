clc
figure, set(gcf, 'Color','white')
set(gca, 'nextplot','replacechildren', 'Visible','off');
global timeee
for k = 1:length(x_traj)
%     figure()
%     h = gca;
%     hold on
    clf
    axes
    axis equal
    hold on
    grid on
    fill(([-.5 .5 .5 -.5 -.5]+x_traj(1,k)), [0 0 .8 .8 0], [0 152/255 153/255])
    
    plot([x_traj(1,k), x_traj(1,k) + l*sin(x_traj(2,k))], [.4, .4+l*cos(x_traj(2,k))], 'k', 'LineWidth', 3)
    
    rectangle('Position',[(x_traj(1,k) + l*sin(x_traj(2,k))-.25*sin(pi/4)),(.4+l*cos(x_traj(2,k))-.25*cos(pi/4)),.35,.35],'Curvature',[1,1],...
        'FaceColor',[150/255 0 0])
    
    xlim([-6 6])
    ylim([-2.5 2.5])
    title(num2str(time(k)))

    MM(k) = getframe(gca);
end

movie(MM,1,length(x_traj)/timeee)