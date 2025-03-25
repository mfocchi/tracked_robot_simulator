function plot_solution(solution,p0, pf, params, DEBUG)

 if nargin < 5
    DEBUG=false;
 end


if (DEBUG)
    

    figure   
    plot(solution.time,-params.omega_w_max*ones(size(solution.omega_l)),'k'); hold on; grid on;
    plot(solution.time,params.omega_w_max*ones(size(solution.omega_l)),'k');
    plot(solution.time,solution.omega_l,'or-');
    plot(solution.time,solution.omega_r,'ob-');
    legend({'min','max','omega_w_l','omega_w_r'});
    ylabel('omega_w','interpreter','none')
         
    figure
    %plots of the states
    subplot(3,1,1)
    plot(solution.time_fine, solution.p_fine(1,:),'r') ; hold on;   grid on; 
    plot(solution.time, solution.p(1,:),'ob') ; hold on;    
    ylabel('X')
    
    subplot(3,1,2)
    plot(solution.time_fine, solution.p_fine(2,:),'r') ; hold on;  grid on;  
    plot(solution.time, solution.p(2,:),'ob') ; hold on;    
    ylabel('Y')
    
    subplot(3,1,3)
    plot(solution.time_fine, solution.p_fine(3,:),'r') ; hold on; grid on;   
    plot(solution.time, solution.p(3,:),'ob') ; hold on;
    ylabel('theta')
  
          
    figure
    %plots of v omega
    subplot(2,1,1)
    plot(solution.time,params.v_max*ones(size(solution.v_input)),'k-'); hold on; grid on;
    plot(solution.time,params.v_min*ones(size(solution.v_input)),'k-');
    plot(solution.time, solution.v_input,'ob-') ; hold on;    
    ylabel('v input')
    
    subplot(2,1,2)
    plot(solution.time,params.omega_max*ones(size(solution.omega_input)),'k-'); hold on; grid on;
    plot(solution.time,params.omega_min*ones(size(solution.omega_input)),'k-');
    plot(solution.time, solution.omega_input,'ob-') ; hold on;    
    ylabel('omega input')
    

    
    if ~strcmp(params.model,'UNICYCLE')
        
        [R, beta_l, beta_r, alpha] = evalSlippage(solution.omega_l, solution.omega_r, params);          
        
        figure
        subplot(3,1,1)
        plot(solution.time, alpha,'ro-') ; hold on;    grid on;
        ylabel('alpha')
        
        subplot(3,1,2)
        plot(solution.time, beta_l,'ro-') ; hold on;    grid on;
        plot(solution.time, beta_r,'bo-') ;    
        legend({'beta_l','beta_r'});
        ylabel('long slippage')
        
        subplot(3,1,3)
        plot(solution.time, log(1+abs(R)),'ro-') ; hold on;    grid on;
        
        ylim([-0 4]);
        ylabel('$log(1+\Vert R \Vert $)','interpreter','latex')
     
        
    
    end
    
end

figure;
%1- plots the optimization curve
plot_curve( solution, p0(:), pf(:), DEBUG);
shg
plot_dubins(p0, pf, params);
