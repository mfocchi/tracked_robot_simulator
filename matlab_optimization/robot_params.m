


params.int_method = 'rk4'; %'eul'  'rk4';
params.N_dyn = 40; %dynamic constraints (number of knowts in the discretization) 
params.int_steps = 10 ;%cast(5,"int64"); %0 means normal intergation
params.num_params = 1; %final time

params.w1= 1;  % minimum time (fundamental!)
params.w2= 1; % smoothing on omega, v  
params.w3= 0; % smoothing on xy_der
params.w4= 0; %smoothing on theta der 

params.gearbox = 34.45;
%params.model = 'UNICYCLE';
%params.model = 'SIDEONLY';
params.model = 'LONGSIDE';


RPM2RADS = 1/60*2*pi;                                                              
params.omega_max = inf; %will be determined by wheelmax speed to have the result I had iun the past with the big curve I need to constraint to 0.4
params.omega_min = -inf;%will be determined by wheelmax speed
params.v_max = 0.4;
params.v_min = 0.4; % if you want optimal control similar to dubins put vmax = vmin and use unicycle model
params.VELOCITY_LIMITS = true; % I need for vmax but I comment the ones on omega cause you have already bounds on wheel_max
params.t_max = 40; %TODO put a check on this
params.slack_target = 0.02;
params.DEBUG_COST = false;
constr_tolerance = 1e-3;
params.dt=0.001; % in matlab is used only to evaluate solution, but it will be sent to c++ to define the discretization for the path generation

params.width = 0.606; % [m]
params.sprocket_radius = 0.0856; % [m]

params.side_slippage_estimation = 'EXP'; %NET EXP
params.friction_coeff=0.4; %0.1, 0.4

if params.friction_coeff == 0.1
    params.omega_w_max = 10; % the decision variables are the wheel of the unycicle not the motors
else 
    params.omega_w_max = 18; % the decision variables are the wheel of the unycicle not the motors
end

% to debug: NN models for NN (will be loaded inside the long/side_slip_model.m)
% alpha_model_forcodegen = loadLearnerForCoder('matlabNN/alpha_model_forcodegen'); 
% beta_l_model_forcodegen = loadLearnerForCoder('matlabNN/beta_l_model_forcodegen'); 
% beta_r_model_forcodegen = loadLearnerForCoder('matlabNN/beta_r_model_forcodegen');  
% alpha = predict(alpha_model_forcodegen, [0.3, 0.4])%-0.0166
% beta_l = predict(beta_l_model_forcodegen, [0.3, 0.4])
% beta_r = predict(beta_r_model_forcodegen, [0.3, 0.4])


% PAPER: simulation high speed (1normal, mid, high slippage)
% params.omega_w_max = 7000 *RPM2RADS/ params.gearbox; % the decision variables are the wheel of the unycicle not the motors, these are 4.5 rad/s                                                                                             
% params.omega_max = 0.8;
% params.omega_min = -0.8;
% params.v_max = 0.8;

%1-normal slippage (this gives a result very similar to NET)
params.side_slip_angle_coefficients_left = [-0.3795,   -3.3784];
params.side_slip_angle_coefficients_right = [0.4587,    3.8471];
params.beta_slip_inner_coefficients_left = [ -0.0579,   -2.4456];
params.beta_slip_outer_coefficients_left =  [  0.0588 ,  -2.6375];
params.beta_slip_inner_coefficients_right = [ -0.0618 ,   3.0089];
params.beta_slip_outer_coefficients_right = [  0.0906,    3.7924];
% 
% % 2-mid slippage
params.side_slip_angle_coefficients_left = [3*-0.3795,  0.5* -3.3784];
params.side_slip_angle_coefficients_right = [3*0.4587,   0.5* 3.8471];
params.beta_slip_inner_coefficients_left = [ 3*-0.0579,  0.5* -2.4456];
params.beta_slip_outer_coefficients_left =  [ 3* 0.0588 ,  0.5*-2.6375];
params.beta_slip_inner_coefficients_right = [ 3*-0.0618 ,   0.5*3.0089];
params.beta_slip_outer_coefficients_right = [3*  0.0906,    0.5*3.7924];

% % 
% % 3-high slippage
params.side_slip_angle_coefficients_left = [4*-0.3795,  0.25* -3.3784];
params.side_slip_angle_coefficients_right = [4*0.4587,   0.25* 3.8471];
params.beta_slip_inner_coefficients_left = [ 4*-0.0579,  0.25* -2.4456];
params.beta_slip_outer_coefficients_left =  [ 4* 0.0588 ,  0.25*-2.6375];
params.beta_slip_inner_coefficients_right = [ 4*-0.0618 ,   0.25*3.0089];
params.beta_slip_outer_coefficients_right = [4*  0.0906,    0.25*3.7924];
% 

