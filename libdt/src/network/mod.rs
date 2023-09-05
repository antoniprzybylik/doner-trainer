use nalgebra::DVector;
use nalgebra::DMatrix;

pub trait Network {
    const PARAMS_CNT: usize;
    const NEURONS_IN: usize;
    const NEURONS_OUT: usize;

    fn new() -> Self;
    fn layers_info() -> &'static str;
    fn eval(p: &[f64], x: DVector<f64>) ->
        DVector<f64>;
    fn forward(&mut self, p: &[f64], x: DVector<f64>) ->
        DVector<f64>;
    fn backward(&mut self, p: &[f64]);
    fn jacobian(&mut self, x: &DVector<f64>) ->
        DMatrix<f64>;
}
