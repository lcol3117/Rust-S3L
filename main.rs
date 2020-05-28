extern crate rand;

use rand::Rng;
use std::mem::drop;

fn main() {
  learning_S3L();
}

fn gen_random_f64() -> f64 {
  let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
  rng.gen::<f64>()
}

struct S3LAgent {
  dims: u32,
  mip: f64,
  j: u32,
  xi_table: Vec<Vec<Vec<f64>>>,
  upsilon: f64
}

fn build_s3l_agent(dims: u32, mip: f64, j: u32) -> S3LAgent {
  let initial_xi_table: Vec<Vec<Vec<f64>>> = Vec::new();
  let initial_upsilon: f64 = 0 as f64;
  S3LAgent {
    dims: dims,
    mip: mip,
    j: j,
    xi_table: initial_xi_table,
    upsilon: initial_upsilon,
  }
}

impl S3LAgent {
  fn update_xi_table(&mut self, policy: Vec<f64>, performance: f64) -> Done {
    let performance_vec: Vec<f64> = vec![performance];
    let new_entry: Vec<Vec<f64>> = vec![policy, performance_vec];
    self.xi_table.push(new_entry);
    Done
  }

  fn select_policy(&mut self) -> Vec<f64> {
    self.calculate_upsilon();
    let mut initial_point: Option<Vec<f64>> = None;
    if self.xi_table.len() > 3 {
      initial_point = Some(self.generate_avoidance_point());
    } else {
      let initial_point_unwrapped = (0..(self.dims as u32))
      .collect::<Vec<u32>>()
      .iter()
      .map(|_x| {gen_random_f64()})
      .collect::<Vec<f64>>();
      initial_point = Some(initial_point_unwrapped);
    }
    let probabilistic_max: Vec<f64> = self.generate_probabilistic_max();
    let vector_delta: Vec<f64> = self.sub_vectors(initial_point.unwrap(), probabilistic_max);
    let scalar: f64 = self.get_vector_delta_scalar();
    let modified_vector_delta: Vec<f64> = self.vector_scalar_multiply(vector_delta, scalar);
    let result: Vec<f64> = self.sum_vectors(initial_point.unwrap(), modified_vector_delta);
    result
  }

  fn calculate_upsilon(&mut self) -> Done {
    let upsilon_result: Option<f64> = None;
    if self.xi_table.len() > 3 {
      let xi_table_max_performance: f64 = self.xi_table
      .iter()
      .fold(std::f64::NEG_INFINITY, |a, x| -> f64 {if x[1][0] > a {x[1][0]} else {a}});
      let intial_result: f64 = xi_table_max_performance / self.mip;
      if intial_result >= 0.8 {
        let base: f64 = 5_f64 * (intial_result - 0.8);
        upsilon_result = Some(0.8 + ((base.powf(self.j as f64)) / 5_f64));
      } else {
        upsilon_result = Some(intial_result);
      }
    } else {
      upsilon_result = Some(0_f64);
    }
    self.upsilon = upsilon_result.unwrap();
    Done
  }

  fn generate_avoidance_point(&self) -> Vec<f64> {
    let initial_vector: Vec<f64> = (0..(self.dims as u32))
    .collect::<Vec<u32>>()
    .iter()
    .map(|_x| {gen_random_f64()})
    .collect::<Vec<f64>>();
    let i_avg_xi_table: f64 = (3_f64 / 4_f64) * self.mip;
    let possible_mins: Vec<Vec<Vec<f64>>> = self.xi_table
    .iter()
    .filter(|x| {x[1][0] < i_avg_xi_table})
    .collect::<Vec<&Vec<Vec<f64>>>>()
    .iter()
    .map(|x| {(*x).clone().to_vec()})
    .collect::<Vec<Vec<Vec<f64>>>>();
    let possible_maxs: Vec<Vec<Vec<f64>>> = self.xi_table
    .iter()
    .filter(|x| {x[1][0] >= i_avg_xi_table})
    .collect::<Vec<&Vec<Vec<f64>>>>()
    .iter()
    .map(|x| {(*x).clone().to_vec()})
    .collect::<Vec<Vec<Vec<f64>>>>();
    let possible_min_vectors: Vec<Vec<f64>> = possible_mins
    .iter()
    .map(|x| {x[0]})
    .collect::<Vec<Vec<f64>>>();
    let possible_max_vectors: Vec<Vec<f64>> = possible_maxs
    .iter()
    .map(|x| {x[0]})
    .collect::<Vec<Vec<f64>>>();
    let mut closest_min: Vec<f64> = possible_min_vectors[0];
    let mut best_min_dist: f64 = std::f64::INFINITY;
    let mut dist: Option<f64> = None;
    for i in possible_min_vectors.iter() {
      dist = Some(self.get_l2_n_dist(closest_min, initial_vector));
      if dist.unwrap() < best_min_dist {
        best_min_dist = dist.unwrap();
        closest_min = i.clone();
      }
    }
    assert!(dist != None);
    let mut closest_max: Vec<f64> = possible_max_vectors[0];
    let mut best_max_dist: f64 = std::f64::INFINITY;
    for i in possible_max_vectors.iter() {
      dist = Some(self.get_l2_n_dist(closest_max, initial_vector));
      if dist.unwrap() < best_max_dist {
        best_max_dist = dist.unwrap();
        closest_max = i.clone();
      }
    }
    assert!(dist != None);
    drop(best_max_dist);
    drop(best_min_dist);
    let ddelta: f64 = self.get_l2_n_dist(closest_min, closest_max);
    let dmin: f64 = self.get_l2_n_dist(initial_vector, closest_min);
    let r: f64 = dmin / ddelta;
    let mut tryagain: Option<bool> = None;
    if r < 0.5 {
      tryagain = Some(gen_random_f64() <= (r * self.upsilon));
    } else {
      tryagain = Some(false);
    }
    let result: Option<Vec<f64>> = None;
    if tryagain.unwrap() {
      drop(tryagain);
      result = Some(self.generate_avoidance_point());
    } else {
      result = Some(initial_vector);
    }
    result.unwrap()
  }

  fn generate_probabilistic_max(&self) -> Vec<f64> {
    let ls: Vec<Vec<Vec<f64>>> = (self.xi_table).clone();
    let ranking: Vec<Vec<Vec<f64>>> = ls
    .sort_by(|a: f64, b: Vec<Vec<f64>>| -> f64 {a[1][0].partial_cmp(&b[1][0]).unwrap()});
    let initial_rr: Vec<Vec<Vec<f64>>> = ranking
    .iter()
    .rev()
    .collect::<Vec<&Vec<Vec<f64>>>>()
    .iter()
    .map(|x| {(*x).clone().to_vec()})
    .collect::<Vec<Vec<Vec<f64>>>>();
    let rr = initial_rr.clone();
    for i in 0..(rr.len() - 1) {
      if gen_random_f64() < (self.upsilon / 2_f64) {
        let tmp: Vec<Vec<f64>> = (rr[i + 1]).clone();
        rr[i + 1] = (rr[i]).clone();
        rr[i] = tmp.clone();
        drop(tmp);
      }
    }
    let newranking: Vec<Vec<Vec<f64>>> = rr.clone();
    let newtop: Vec<f64> = newranking[0][0];
    newtop.clone()
  }

  fn get_vector_delta_scalar(&self) -> f64 {
    (self.upsilon).powf(self.j as f64)
  }

  fn sub_vectors(&self, a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    let region_point: Vec<f64> = b.clone();
    let d_range: Vec<usize> = (0..(self.dims))
    .into_iter()
    .map(|x: u32| -> usize {x as usize})
    .collect::<Vec<usize>>();
    let deltas: Vec<f64> = d_range
    .iter()
    .map(|x: &usize| -> f64 {region_point[*x] - a[*x]})
    .collect::<Vec<f64>>();
    drop(d_range);
    deltas
  }

  fn vector_scalar_multiply(&self, m_vector: Vec<f64>, m_scalar: f64) -> Vec<f64> {
    m_vector
    .iter()
    .map(|x| {x * m_scalar})
    .collect::<Vec<f64>>()
  }

  fn sum_vectors(&self, a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    let d_range: Vec<usize> = (0..(self.dims))
    .into_iter()
    .map(|x: u32| -> usize {x as usize})
    .collect::<Vec<usize>>();
    let sum_result: Vec<f64> = d_range
    .iter()
    .map(|x: &usize| -> f64 {a[*x] + b[*x]})
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
    .collect::<Vec<(&f64,&f64)>>()
    .iter()
    .fold(0_f64, |a, x| {a + (((*x).0 - (*x).1).powf(2))});
    let dist: f64 = dist_sum.sqrt();
    dist
  }
}

struct Done;

fn real_learn_step(agentmodel: S3LAgent) -> Vec<Vec<f64>> {
  let policy: Vec<f64> = agentmodel.select_policy();
  let performance: f64 = evaluate_policy(policy);
  agentmodel.update_xi_table(policy, performance);
  println!("Agent tried policy {:?}. ", policy);
  println!("This lead to performance {:?}. ", performance);
  let current_performance_vec: Vec<f64> = vec![performance];
  let pp_vec: Vec<Vec<f64>> = vec![policy, current_performance_vec];
  pp_vec
}

fn evaluate_policy(policy: Vec<f64>) -> f64 {
  let real_max: Vec<f64> = vec![0.7, 0.7];
  let real_max_dist: f64 = get_l2_n_dist(policy, real_max);
  let initial: f64 = (2_f64 - real_max_dist) + 1_f64;
  let local_max: Vec<f64> = vec![0.2,0.2];
  let local_max_dist: f64 = get_l2_n_dist(policy, local_max);
  let other_inital: f64 = 2_f64 - local_max_dist;
  let mut policy_result: Option<f64> = None;
  if real_max_dist < local_max_dist {
    policy_result = Some(initial);
  } else {
    policy_result = Some(other_inital);
  }
  policy_result.unwrap()
}

fn get_l2_n_dist(a: Vec<f64>, b: Vec<f64>) -> f64 {
  let z = a
  .iter()
  .zip(b.iter());
  let dist_sum: f64 = z
  .into_iter()
  .collect::<Vec<(&f64,&f64)>>()
  .iter()
  .fold(0_f64, |a, x| {a + (((*x).0 - (*x).1).powf(2))});
  let dist: f64 = dist_sum.sqrt();
  dist
}

fn pure_exploration_step(agentmodel: S3LAgent) -> Done {
  let policy = (0..(agentmodel.dims as u32))
  .collect::<Vec<u32>>()
  .iter()
  .map(|_x| {gen_random_f64()})
  .collect::<Vec<f64>>();
  let performance = evaluate_policy(policy);
  agentmodel.update_xi_table(policy, performance);
  println!("Agent explored policy {:?}. ", policy);
  Done
}

fn learning_S3L() -> Done {
  let mut agentmodel: S3LAgent = build_s3l_agent(2, 3.1, 2);
  for _i in 0..3 {
    pure_exploration_step(agentmodel);
  }
  let mut not_done: bool = true;
  let mut cnt: u32 = 2;
  let mut best_policy: Vec<f64> = Vec::new();
  let mut best_performance: f64 = std::f64::NEG_INFINITY;
  let mut best_performance_vec: Vec<f64> = vec![best_performance];
  let mut best_xi_table_entry: Vec<Vec<f64>> = vec![best_policy, best_performance_vec];
  while not_done {
    real_learn_step(agentmodel);
    best_xi_table_entry = agentmodel.xi_table
    .clone()
    .iter()
    .max_by(|a, b| {a[1][0].cmp(b[1][0]).unwrap()});
    best_policy = best_xi_table_entry[0];
    best_performance = best_xi_table_entry[1][0];
    not_done = best_performance < 2.9;
    println!("Done state is {:?}. ", !not_done);
    println!("{:?} steps have passed. ", cnt);
    cnt = cnt + 1;
  }
  println!("==============DONE==============");
  println!("The agent decided on policy {:?}. ", best_policy);
  println!("This policy had performance {:?}. ", best_performance);
  Done
}
