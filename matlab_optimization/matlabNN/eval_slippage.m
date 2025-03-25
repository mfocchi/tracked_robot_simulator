clc
omega_l = 6.4;
omega_r= -0.4;


alpha_model_forcodegen = loadLearnerForCoder('alpha_model_forcodegen');

% Load the model only once
beta_l_model_forcodegen = loadLearnerForCoder('beta_l_model_forcodegen');

% Load the model only once
beta_r_model_forcodegen = loadLearnerForCoder('beta_r_model_forcodegen');



%%FAST (0.0020s) use martlab model trained with regressorLearner
alpha = predict(alpha_model_forcodegen, [omega_l, omega_r])

%%FAST (0.0020s) use martlab model trained with regressorLearner
beta_l = predict(beta_l_model_forcodegen, [omega_l, omega_r])

%%FAST (0.0020s) use martlab model trained with regressorLearner
beta_r = predict(beta_r_model_forcodegen, [omega_l, omega_r])