use nalgebra::SVector;
use nalgebra::SMatrix;

pub trait Network<const PARAMS_CNT: usize,
                  const NEURONS_IN: usize,
                  const NEURONS_OUT: usize>
{
    const PARAMS_CNT: usize = PARAMS_CNT;
    const NEURONS_IN: usize = NEURONS_IN;
    const NEURONS_OUT: usize = NEURONS_OUT;

    fn new() -> Self;
    fn layers_info() -> &'static str;
    fn eval(p: &[f64],
            x: SVector<f64, NEURONS_IN>) ->
        SVector<f64, NEURONS_OUT>;
    fn forward(&mut self, p: &[f64],
               x: SVector<f64, NEURONS_IN>) ->
        SVector<f64, NEURONS_OUT>;
    fn backward(&mut self, p: &[f64]);
    fn jacobian(&mut self,
                x: &SVector<f64, NEURONS_IN>) ->
        SMatrix<f64, NEURONS_OUT,
                PARAMS_CNT>;
}
