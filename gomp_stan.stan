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

  real mu_a;                            // global log-mean of a
  real<lower=0> tau_a;                  // std dev for log(a)
  vector[N] z_a;                        // standard normal latent variables
}

transformed parameters {
  vector[N] a;
  vector[N] b;
  vector[N] p;

  a = exp(mu_a + tau_a * z_a);
  for (i in 1:N) {
    if (a[i] == 0) a[i] = 0.0001;       // ensure 'a' is not exactly zero
  }

  b = exp(to_vector(mu1) + var1 * beta);
  p = 1 - exp(-b ./ a .* (exp(a .* dt) - 1));
}

model {
  // priors
  mu1 ~ normal(0, 1);
  beta ~ normal(0, 1);

  mu_a ~ normal(0, 1);
  tau_a ~ normal(0, 1);
  z_a ~ normal(0, 1);

  // likelihood
  r ~ binomial(n, p);
}

generated quantities {
  array[N] int r_new;
  vector[N] p_new;
  vector[N] a_out;
  vector[N] b_out;

  p_new = 1 - exp(-b ./ a .* (exp(a .* dt) - 1));
  for (i in 1:N) {
    r_new[i] = binomial_rng(n[i], p_new[i]);
    a_out[i] = a[i];
    b_out[i] = b[i];
  }
  real rate = mean(exp(to_vector(mu1)));
}
