use nalgebra::SVector;
use nalgebra::RowSVector;

use super::network::Network;

/// Neural network trainer.
pub trait Trainer<N,
              const PARAMS_CNT: usize,
              const NEURONS_IN: usize,
              const NEURONS_OUT: usize>
where
    N: Network<PARAMS_CNT, NEURONS_IN, NEURONS_OUT>
{
    fn new(nn: N, p: Vec<f64>,
           x_values: Vec<SVector<f64, NEURONS_IN>>,
           d_values: Vec<SVector<f64, NEURONS_OUT>>) -> Self;
    fn make_step(&mut self);

    fn cost(&self) -> f64;
    fn grad(&mut self) -> RowSVector<f64, PARAMS_CNT>;
    fn grad_norm(&mut self) -> f64;
    fn params(&self) -> &[f64];
}

mod common;

mod gd_trainer;
pub use gd_trainer::*;

mod cg_trainer;
pub use cg_trainer::*;
