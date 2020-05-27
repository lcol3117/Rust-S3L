struct S3L_agent {
  dims: u32,
  mip: f64,
  j: u32,
  xi_table: Vec<xi_table_entry>,
  upsilon: f64
}

fn build_S3L_agent(dims: u32, mip: f64, j: u32) -> S3L_agent {
  let initial_xi_table: Vec<Vec<f64>> = Vec::new();
  let initial_upsilon: f64 = 0 as f64;
  S3L_agent {
    dims: dims,
    mip: mip,
    j: j,
    xi_table: initial_xi_table,
    upsilon: initial_upsilon,
  }
}

impl S3L_agent {
  fn update_xi_table(&mut self, policy: Vec<f64>, performance: f64) -> Done {
    let performance_vec: Vec<f64> = vec![performance];
    let new_entry: xi_table_entry = vec![policy, performance_vec];
    self.xi_table.push(new_entry);
    Done
  }

  fn selectPolicy(&mut self) -> Vec<f64> {
    self.calculate_upsilon();
    if self.xi_table.len() > 3 {
      let initial_point: Vec<f64> = self.generate_avoidance_point();
    } else {
      let initial_point: Vec<f64> = (0..(self.dims as u32))
      .collect::<Vec<u32>>()
      .iter()
      .map(|_x| {rng.gen::<f64>()})
      .collect::<Vec<f64>>();
    }
    let probabilistic_max: Vec<f64> = self.generate_probabilistic_max();
    let vector_delta: Vec<f64> = self.sub_vectors(initial_point, probabilistic_max);
    let scalar: f64 = self.get_vector_delta_scalar();
    let modified_vector_delta: Vec<f64> = self.vector_scalar_multiply(vector_delta, scalar);
    let result: Vec<f64> = self.sum_vectors(initial_point, modified_vector_delta);
    result
  }

  fn calculate_upsilon(&mut self) -> Done {
    if self.xi_table.len() > 3 {
      let xi_table_max_performance: f64 = self.xi_table
      .iter()
      .fold(f64::NEG_INFINITY, |a: f64, x: f64| -> f64 {if x[1][0] > a {x[1][0]} else {a}}});
      let intial_result: f64 = xi_table_max_performance / self.mip;
      if intial_result >= 0.8 {
        let base: f64 = (5_f64 * (intial_result - 0.8));
        let upsilon_result: f64 = 0.8 + ((base.pow(self.j)) / 5_f64);
      } else {
        let upsilon_result: f64 = intial_result;
      }
    } else {
      let upsilon_result: f64 = 0_f64;
    }
    self.upsilon = upsilon_result;
    Done
  }

  fn generate_avoidance_point(&self) -> Vec<f64> {
    let initial_vector: Vec<f64> = (0..(self.dims as u32))
    .collect::<Vec<u32>>()
    .iter()
    .map(|_x| {rng.gen::<f64>()})
    .collect::<Vec<f64>>();
    let i_avg_xi_table: f64 = (3_f64 / 4_f64) * self.mip;
    let possible_mins: Vec<xi_table_entry> = self.xi_table
    .iter()
    .filter(|x| {x[1][0] < i_avg_xi_table})
    .collect::<Vec<xi_table_entry>>();
    let possible_maxs: Vec<xi_table_entry> = self.xi_table
    .iter()
    .filter(|x| {x[1][0] >= i_avg_xi_table})
    .collect::<Vec<xi_table_entry>>();
    let possible_min_vectors: Vec<Vec<f64>> = possible_mins
    .iter()
    .map(|x| {x[0]})
    .collect::<Vec<Vec<f64>>>();
    let possible_max_vectors: Vec<Vec<f64>> = possible_maxs
    .iter()
    .map(|x| {x[0]})
    .collect::<Vec<Vec<f64>>>();
    let mut closest_min: Vec<f64> = possible_min_vectors[0];
    let mut best_min_dist: f64 = f64::INFINITY;
    for i in possible_min_vectors.iter() {
      dist = self.get_l2_n_dist(closest_min, initial_vector);
      if dist < best_min_dist {
        best_min_dist = dist;
        closest_min = (&i).clone();
      }
    }
    let mut closest_max: Vec<f64> = possible_max_vectors[0];
    let mut best_max_dist: f64 = f64::INFINITY;
    for i in possible_max_vectors.iter() {
      dist = self.get_l2_n_dist(closest_max, initial_vector);
      if dist < best_max_dist {
        best_max_dist = dist;
        closest_max = (&i).clone();
      }
    }
    drop(best_max_dist);
    drop(best_min_dist);
    let ddelta: f64 = self.get_l2_n_dist(closest_min, closest_max);
    let dmin: f64 = self.get_l2_n_dist(initial_vector, closest_min);
    let r: f64 = dmin / ddelta;
    if r < 0.5 {
      let tryagain: bool = rng.gen::<f64>() <= (r * self.upsilon);
    } else {
      let tryagain: bool = false;
    }
    if tryagain {
      drop(tryagain);
      let result: Vec<f64> = self.generate_avoidance_point();
    } else {
      let result: Vec<f64> = initial_vector;
    }
    result
  }

  fn generate_probabilistic_max(&self) -> Vec<f64> {
    let ls: Vec<xi_table_entry> = (&(self.xi_table)).clone();
    let ranking: Vec<xi_table_entry> = ls
    .sort_by(|a, b| {a[1][0].partial_cmp(b[1][0]).unwrap()});
    let initial_rr: Vec<xi_table_entry> = ranking
    .iter()
    .rev()
    .collect::<Vec<xi_table_entry>>();
    let rr = (&initial_rr).clone();
    for i in 0..(rr.len() - 1) {
      if (rng.gen::<f64>() < (self.upsilon / 2)) {
        let tmp: xi_table_entry = (&rr[i + 1]).clone();
        rr[i + 1] = (&rr[i]).clone();
        rr[i] = (&tmp).clone();
        drop(tmp);
      }
    }
    let newranking: Vec<xi_table_entry> = (&rr).clone();
    let newtop: Vec<f64> = newranking[0][0];
    (&newtop).clone()
  }

  fn get_vector_delta_scalar(&self) -> f64 {
    (self.upsilon).pow(self.j)
  }

  fn sub_vectors(&self, a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    let region_point: Vec<f64> = (&b).clone();
    let d_range: Vec<usize> = 0..(self.dims)
    .iter()
    .map(|x: u32| -> usize {x as usize})
    .collect::<Vec<usize>>();
    let deltas: Vec<f64> = d_range
    .iter()
    .map(|x: usize| -> f64 {region_point[x] - a[x]})
    .collect::<Vec<f64>>();
    drop(d_range);
    deltas
  }

  fn vector_scalar_multiply(&self, m_vector: Vec<f64>, m_scalar: f64) -> Vec<f64> {
    m_vector
    .iter()
    .map(|x| {x * m_scalar})
    .collect::<Vec<f64>>();
  }

  fn sum_vectors(&self, a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    let d_range: Vec<usize> = 0..(self.dims)
    .iter()
    .map(|x: u32| -> usize {x as usize})
    .collect::<Vec<usize>>();
    let sum_result: Vec<f64> = d_range
    .iter()
    .map(|x: usize| -> f64 {a[x] + b[x]})
    .collect::<Vec<f64>>();
    drop(d_range);
    sum_result
  }

  fn get_l2_n_dist(&self, a: Vec<f64>, b: Vec<f64>) -> f64 {
    let z = a
    .iter()
    .zip(b.iter());
    let dist_sum: f64 = z
    .into_iter()
    .iter()
    .fold(0_f64, |a, x| {a + ((x.0 - x.1).pow(2))});
    let dist: f64 = dist_sum.sqrt();
    dist
  }
}

struct Done;