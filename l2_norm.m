function vec_l2_norm = l2_norm(vec)
% L2-normalization of a vector
% Jay Chakravarty 
% April 2015
norm_vec = norm(vec)+eps;
vec_l2_norm = vec./norm_vec;

end
