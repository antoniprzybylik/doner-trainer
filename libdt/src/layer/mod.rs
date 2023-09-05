use nalgebra as na;

use na::DMatrix;
use na::DVector;

/// Neural network layer.
pub trait Layer {
    const PARAMS_CNT: usize;
    const NEURONS_IN: usize;
    const NEURONS_OUT: usize;

    unsafe fn eval_unchecked(p: &[f64], x: DVector<f64>) -> DVector<f64>;
    fn eval(p: &[f64], x: DVector<f64>) -> DVector<f64>;
    fn forward(&mut self, p: &[f64], x: DVector<f64>) -> DVector<f64>;
    fn backward(&mut self, p: &[f64]);
    fn chain_element(&self) -> &DMatrix<f64>;
    fn chain_end(&self, x: &DVector<f64>) -> DMatrix<f64>;
}

mod lin_layer;
pub use lin_layer::*;

mod sigma_layer;
pub use sigma_layer::*;

mod gelu_layer;
pub use gelu_layer::*;

mod softmax_layer;
pub use softmax_layer::*;
