use nalgebra as na;

use na::SMatrix;
use na::SVector;

/// Warstwa o określonej liczbie neuronów wejściowych i wyjściowych.
pub trait Layer<const NEURONS_IN: usize, const NEURONS_OUT: usize> {
    const PARAMS_CNT: usize;
    const NEURONS_IN: usize = NEURONS_IN;
    const NEURONS_OUT: usize = NEURONS_OUT;

    unsafe fn eval_unchecked(p: &[f64], x: SVector<f64, NEURONS_IN>) -> SVector<f64, NEURONS_OUT>;
    fn eval(p: &[f64], x: SVector<f64, NEURONS_IN>) -> SVector<f64, NEURONS_OUT>;
    fn forward(&mut self, p: &[f64], x: SVector<f64, NEURONS_IN>) -> SVector<f64, NEURONS_OUT>;
    fn backward(&mut self, p: &[f64]);
    fn chain_element(&self) -> &SMatrix<f64, NEURONS_OUT, NEURONS_IN>;
    fn chain_end(&self, x: &SVector<f64, NEURONS_IN>) -> SMatrix<f64, NEURONS_OUT, { Self::PARAMS_CNT }>;
}

mod lin_layer;
pub use lin_layer::*;

mod sigma_layer;
pub use sigma_layer::*;
