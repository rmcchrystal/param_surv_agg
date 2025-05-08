data {
  int<lower=0> N;                       // number of data points
  array[N] int<lower=0> r;              // observed counts
  array[N] int<lower=0> n;              // number of trials
  vector[N] dt;                         // time intervals
  vector[N] var1;                       // covariate
}

parameters {
  array[N] real mu1;                    // intercepts per trial
  real beta;                            // coefficient
  real a_raw;                           // raw parameter for 'a'
}

transformed parameters {
  real a;
  vector[N] b;
  vector[N] p;

  a = a_raw == 0 ? 0.0001 : a_raw;      // ensure 'a' is not exactly zero
  b = exp(to_vector(mu1) + var1 * beta);
  p = 1 - exp(-b / a .* (exp(a * dt) - 1));
}

model {
  // priors
  mu1 ~ normal(0, 1);
  beta ~ normal(0, 1);
  a_raw ~ normal(0, 1);

  // likelihood
  r ~ binomial(n, p);
}

generated quantities {
  array[N] int r_new;
  vector[N] p_new;

  p_new = 1 - exp(-b / a .* (exp(a * dt) - 1));
  for (i in 1:N) {
    r_new[i] = binomial_rng(n[i], p_new[i]);
  }
  real rate = mean(exp(to_vector(mu1)));
}
