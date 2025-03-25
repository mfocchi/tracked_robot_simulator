
function solution = optimize_cpp(p0,  pf, omega_l0, omega_r0, t0, params) 
    %make it column vector
    p0 = p0(:);
    pf = pf(:);
       
    % unfortunately that needs to be fixed and cannot be parameters for code generation
    constr_tolerance = 1e-3;
    max_func_evaluations = 5000;
    max_iterations = 500;


    [v_input,omega_input] =  computeVelocitiesFromTracks(omega_l0, omega_r0, params);
    if  any(v_input >params.v_max) 
        disp('initialization is unfeasible')
    end
    
    x0 = [ t0 ,  omega_l0,  omega_r0];
     
    lb = [ 0  , -params.omega_w_max*ones(1,length(omega_l0)),  -params.omega_w_max*ones(1,length(omega_r0))];
    ub = [ params.t_max, params.omega_w_max*ones(1,length(omega_l0)),  params.omega_w_max*ones(1,length(omega_r0))];
    options = optimoptions('fmincon','Display','iter','Algorithm','sqp',  ... % does not always satisfy bounds
    'MaxFunctionEvaluations',  max_func_evaluations, 'ConstraintTolerance',constr_tolerance, ...
    'MaxIterations', max_iterations);

    tic
    [x, final_cost, EXITFLAG, output] = fmincon(@(x) cost(x, p0,  pf, params), x0,[],[],[],[],lb,ub,  @(x) constraints(x, p0,  pf, params) , options);
    toc

    

    solution = eval_solution(x, params.dt,  p0, pf, params) ;
    solution.x = x;
    solution.cost = final_cost;
    solution.problem_solved = EXITFLAG ;%(EXITFLAG == 1) || (EXITFLAG == 2);
    % 1 First-order optimality measure was less than options.OptimalityTolerance, and maximum constraint violation was less than options.ConstraintTolerance.
    % 0 Number of iterations exceeded options.MaxIterations or number of function evaluations exceeded options.MaxFunctionEvaluations.
    % -1 Stopped by an output function or plot function.
    % -2 No feasible point was found.
    % 2 Change in x was less than options.StepTolerance (Termination tolerance on x, a scalar, the default is 1e-10) and maximum constraint violation was less than options.ConstraintTolerance.


    solution.optim_output = output;
    % evaluate constraint violation 
    [c ceq, solution_constr] = constraints(x, p0,  pf, params);
    solution.c = c;    
    solution.solution_constr = solution_constr;
   
 
end

