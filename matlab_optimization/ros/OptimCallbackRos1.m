function resp = OptimCallbackRos1(~, req,resp)


    global params  
    %clc
    close all
    
   

    p0(1) = req.X0;
    p0(2) = req.Y0;
    p0(3) = req.Theta0;
    
    pf(1) = req.Xf;
    pf(2) = req.Yf;
    pf(3) = req.Thetaf;
    params.v_max = req.Vmax; %overwrite default with user one
    plan_type = req.PlanType;
    

    %make them columns
    p0 = p0(:);
    pf = pf(:);
    
    fprintf(2,"NEW TARGET:%f %f %f \n", pf);

    %check if omega is feasible
    [feas_omega_min, feas_omega_max] = computeFeasibleOmega(params.v_max, params);
    if (params.omega_max)>feas_omega_max
        fprintf(2, "OMEGA IN DUBINS IS BEYOND THE LIMITS setting to:  %f\n",feas_omega_max );
        params.omega_max = feas_omega_max;
        params.omega_min = -feas_omega_max;
    end

    dubConnObj = dubinsConnection;
    curvature_max = params.omega_max/params.v_max;
    dubConnObj.MinTurningRadius = 1/curvature_max;
    
    [pathSegObj, pathCosts] = connect(dubConnObj,p0',pf');
    %show(pathSegObj{1})
    % get total time
    t0 = sum(pathSegObj{1}.MotionLengths)/params.v_max;

    % compute the omega from dubin
    omegas = get_omega_from_dubins(pathSegObj{1}, params.v_max, 1/curvature_max);
    %map to wheel omega
    [omega_l0, omega_r0, t_rough] = getWheelVelocityParamsFromDubin(params, pathSegObj{1}.MotionLengths, omegas);

    if strcmp(plan_type, "dubins")
        tic
        %generate inputs from dubins om a fome grid (dt)
        [omega_l_fine, omega_r_fine, ] = getWheelVelocityParamsFromDubin(params, pathSegObj{1}.MotionLengths, omegas, params.dt);
        %integrate dubin on a fine grid (dt)
        params.int_steps = 0;
        params.model = 'UNICYCLE'; %need to set unicycle!!!
        %integrate the fine grid omegas
        [states, t] = computeRollout(p0, 0,params.dt, length(omega_l_fine), omega_l_fine, omega_r_fine, params);   
        [v_input,omega_input] = computeVelocitiesFromTracks(omega_l_fine, omega_r_fine, params);

        resp.DesX = states(1,:);
        resp.DesY = states(2,:);
        resp.DesTheta = states(3,:);
        resp.DesV = v_input;
        resp.DesOmega = omega_input;
        resp.Dt = params.dt;
        

        %plot_dubins(p0, pf, params);
        fprintf(2,"NEW dubins\n")
        disp('Duration Tf')
        Tf = sum(pathSegObj{1}.MotionLengths)/params.v_max;
        resp.Tf = Tf;
        toc
        plot_dubins(p0, pf, params);  
    elseif strcmp(plan_type, "optim")    
        
        solution = optimize_cpp_mex(p0,  pf, omega_l0, omega_r0, t0,  params); 
        %plot_solution(solution,p0, pf, params, false);
        %these vectors are too big to transfer to the robot TODO if you fix
        %C++  side then send these otherwise you can have integration
        %errors
         
        resp.DesX = solution.p_fine(1,:);
        resp.DesY = solution.p_fine(2,:);
        resp.DesTheta = solution.p_fine(3,:);
        resp.DesV = solution.v_input_fine;
        resp.DesOmega = solution.omega_input_fine;
        resp.Dt = params.dt;
        resp.Tf = solution.Tf;

        fprintf(2,"NEW OPTIM\n")
        %solution.Tf
        %length(resp.des_theta)
        disp('Duration Tf')
        solution.Tf
        disp('Achieved targetcl')
        solution.achieved_target
        plot_solution(solution,p0, pf, params, false);   
   
    else
        disp("wrong plan type")
    end
    %
    % req.plan_type
    % resp.des_x(end-10:end)
    % resp.des_y(end-10:end)
    % resp.des_theta(end-10:end)
    % resp.des_v(end-10:end)
    % resp.des_omega(end-10:end)
    % resp.dt
    
end