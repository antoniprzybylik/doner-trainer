use nalgebra::DVector;
use nalgebra::RowDVector;

use super::network::Network;

/// Neural network trainer.
pub trait Trainer<N: Network> {
    fn new(nn: N, p: Vec<f64>,
           x_values: Vec<DVector<f64>>,
           d_values: Vec<DVector<f64>>) -> Self;
    fn make_step(&mut self);

    fn cost(&self) -> f64;
    fn grad(&mut self) -> RowDVector<f64>;
    fn grad_norm(&mut self) -> f64;
    fn params(&self) -> &[f64];
}

mod common;

mod gd_trainer;
pub use gd_trainer::*;

mod cg_trainer;
pub use cg_trainer::*;

mod lm_trainer;
pub use lm_trainer::*;
