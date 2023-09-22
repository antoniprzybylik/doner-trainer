use nalgebra as na;

use na::DMatrix;
use na::DVector;
use na::Matrix;
use na::Vector;
use na::MatrixView;
use na::base::dimension as dim;
use na::base::VecStorage;

use super::Layer;

pub struct SumLayer<const NEURONS_IN: usize, const NEURONS_OUT: usize> {
    pub signal: Vector::<f64, dim::Dyn,
                         VecStorage::<f64, dim::Dyn, dim::U1>>,
    chain_element: Matrix::<f64, dim::Dyn, dim::Dyn,
                            VecStorage::<f64, dim::Dyn, dim::Dyn>>,
}

impl<const NEURONS_IN: usize,
     const NEURONS_OUT: usize> SumLayer<NEURONS_IN, NEURONS_OUT>
{
    pub fn new() -> Self {
        Self {
            signal: Vector::from_element_generic(dim::Dyn(Self::NEURONS_OUT),
                                                 dim::U1, 0f64),
            chain_element: Matrix::from_element_generic(dim::Dyn(Self::NEURONS_OUT),
                                                        dim::Dyn(Self::NEURONS_IN), 0f64),
        }
    }
}

impl<const NEURONS_IN: usize, const NEURONS_OUT: usize> Layer
    for SumLayer<NEURONS_IN, NEURONS_OUT>
{
    const PARAMS_CNT: usize = NEURONS_IN * NEURONS_OUT;
    const NEURONS_IN: usize = NEURONS_IN;
    const NEURONS_OUT: usize = NEURONS_OUT;

    unsafe fn eval_unchecked(p: &[f64], x: DVector<f64>) -> DVector<f64> {
        let m = MatrixView::from_slice_generic_unchecked(
            p,
            0,
            dim::Dyn(NEURONS_OUT),
            dim::Dyn(NEURONS_IN),
        );

        &m * x
    }

    fn eval(p: &[f64], x: DVector<f64>) -> DVector<f64> {
        assert_eq!(p.len(), Self::PARAMS_CNT);
        assert_eq!(x.len(), Self::NEURONS_IN);

        let m = MatrixView::from_slice_generic(
            p, dim::Dyn(NEURONS_OUT), dim::Dyn(NEURONS_IN));

        &m * x
    }

    fn forward(&mut self, p: &[f64], x: DVector<f64>) -> DVector<f64> {
        assert_eq!(p.len(), Self::PARAMS_CNT);
        assert_eq!(x.len(), Self::NEURONS_IN);

        self.signal = Self::eval(p, x);
        self.signal.clone()
    }

    fn backward(&mut self, p: &[f64]) {
        self.chain_element = MatrixView::from_slice_generic(
            p, dim::Dyn(NEURONS_OUT), dim::Dyn(NEURONS_IN)).into();
    }

    fn chain_element(&self) -> &DMatrix<f64> {
        &self.chain_element
    }

    fn chain_end(&self, x: &DVector<f64>) -> DMatrix<f64>
    {
        let mut matrix: DMatrix<f64> =
            DMatrix::from_element_generic(
            dim::Dyn(Self::NEURONS_OUT),
            dim::Dyn(Self::PARAMS_CNT), 0f64);

        for i in 0..NEURONS_IN {
            for j in 0..NEURONS_OUT {
                matrix[(j, i*NEURONS_OUT+j)] = x[i];
            }
        }

        matrix
    }

    fn default_initial_params() -> Vec<f64> {
        let mut p: Vec<f64> =
            Vec::with_capacity(Self::PARAMS_CNT);
        for _ in 0..Self::PARAMS_CNT {
            p.push(1f64);
        }

        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval() {
        let p: [f64; 4] = [1., 1., 1., 0.];
        let x = DVector::from_column_slice(
            na::vector![1f64, 2f64].as_slice());
        
        let y = SumLayer::<2, 2>::eval(&p, x);
        assert_eq!(y, na::vector![3., 1.]);
    }

    #[test]
    fn test_forward() {
        let p: [f64; 4] = [1., 1., 1., 0.];
        let x = DVector::from_column_slice(
            na::vector![1f64, 2f64].as_slice());
        let mut layer = SumLayer::<2, 2>::new();
        
        let y = layer.forward(&p, x);
        assert_eq!(y, na::vector![3., 1.]);
    }

    #[test]
    fn test_backward() {
        let p: [f64; 4] = [1., 1., 1., 0.];
        let x = DVector::from_column_slice(
            na::vector![1f64, 2f64].as_slice());
        let mut layer = SumLayer::<2, 2>::new();
        
        let _ = layer.forward(&p, x);
        layer.backward(&p);

        assert_eq!(*layer.chain_element(),
                   na::matrix![1., 1.;
                               1., 0.]);
    }

    #[test]
    fn test_chain_end() {
        let x = DVector::from_column_slice(
            na::vector![1f64, 2f64].as_slice());
        let layer = SumLayer::<2, 3>::new();

        assert_eq!(layer.chain_end(&x),
                   na::matrix![1., 0., 0., 2., 0., 0.;
                               0., 1., 0., 0., 2., 0.;
                               0., 0., 1., 0., 0., 2.]);
    }
}
