use nalgebra as na;

use na::Const;
use na::MatrixView;
use na::SMatrix;
use na::SVector;

use super::Layer;

pub struct LinLayer<const NEURONS_IN: usize, const NEURONS_OUT: usize> {
    pub signal: SVector<f64, NEURONS_OUT>,
    chain_element: SMatrix<f64, NEURONS_OUT, NEURONS_IN>,
}

impl<const NEURONS_IN: usize,
     const NEURONS_OUT: usize> LinLayer<NEURONS_IN, NEURONS_OUT>
{
    pub fn new() -> Self {
        Self {
            signal: na::zero(),
            chain_element: na::zero(),
        }
    }
}

impl<const NEURONS_IN: usize, const NEURONS_OUT: usize> Layer<NEURONS_IN, NEURONS_OUT>
    for LinLayer<NEURONS_IN, NEURONS_OUT>
{
    const PARAMS_CNT: usize = NEURONS_IN * NEURONS_OUT + NEURONS_OUT;

    unsafe fn eval_unchecked(p: &[f64], x: SVector<f64, NEURONS_IN>) -> SVector<f64, NEURONS_OUT> {
        let m = MatrixView::from_slice_generic_unchecked(
            p,
            0,
            Const::<NEURONS_OUT>,
            Const::<NEURONS_IN>,
        );
        let v = MatrixView::from_slice_generic_unchecked(
            p,
            NEURONS_OUT * NEURONS_IN,
            Const::<NEURONS_OUT>,
            Const::<1>,
        );

        &(&m * x) + v
    }

    fn eval(p: &[f64], x: SVector<f64, NEURONS_IN>) -> SVector<f64, NEURONS_OUT> {
        let m = MatrixView::from_slice_generic(p, Const::<NEURONS_OUT>, Const::<NEURONS_IN>);
        let v = MatrixView::from_slice_generic(
            &p[NEURONS_OUT * NEURONS_IN..],
            Const::<NEURONS_OUT>,
            Const::<1>,
        );

        &(&m * x) + v
    }

    fn forward(&mut self, p: &[f64], x: SVector<f64, NEURONS_IN>) -> SVector<f64, NEURONS_OUT> {
        self.signal = Self::eval(p, x);

        self.signal.clone()
    }

    fn backward(&mut self, p: &[f64]) {
        self.chain_element =
            MatrixView::from_slice_generic(p, Const::<NEURONS_OUT>, Const::<NEURONS_IN>).into();
    }

    fn chain_element(&self) -> &SMatrix<f64, NEURONS_OUT, NEURONS_IN> {
        &self.chain_element
    }

    fn chain_end(&self, x: &SVector<f64, NEURONS_IN>) ->
        SMatrix<f64, NEURONS_OUT, { Self::PARAMS_CNT }>
    {
        let mut matrix: SMatrix<f64, NEURONS_OUT, { Self::PARAMS_CNT }> = na::zero();
        for i in 0..NEURONS_IN*NEURONS_OUT {
            matrix[(i.div_euclid(NEURONS_IN), i)] = x[i.rem_euclid(NEURONS_IN)];
        }
        for i in NEURONS_IN*NEURONS_OUT..Self::PARAMS_CNT {
            matrix[(i - NEURONS_IN*NEURONS_OUT, i)] = 1.;           
        }

        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval() {
        let p: [f64; 6] = [1., 1., 1., 0., 7., 7.];
        let x = na::vector![1., 2.];
        
        let y = LinLayer::<2, 2>::eval(&p, x);
        assert_eq!(y, na::vector![10., 8.]);
    }

    #[test]
    fn test_forward() {
        let p: [f64; 6] = [1., 1., 1., 0., 7., 7.];
        let x = na::vector![1., 2.];
        let mut layer = LinLayer::<2, 2>::new();
        
        let y = layer.forward(&p, x);
        assert_eq!(y, na::vector![10., 8.]);
    }

    #[test]
    fn test_backward() {
        let p: [f64; 6] = [1., 1., 1., 0., 7., 7.];
        let x = na::vector![1., 2.];
        let mut layer = LinLayer::<2, 2>::new();
        
        let _ = layer.forward(&p, x);
        layer.backward(&p);

        assert_eq!(*layer.chain_element(),
                   na::matrix![1., 1.;
                               1., 0.]);
    }

    #[test]
    fn test_chain_end() {
        let x = na::vector![1f64, 2f64];
        let layer = LinLayer::<2, 3>::new();

        assert_eq!(layer.chain_end(&x),
                   na::matrix![1., 2., 0., 0., 0., 0., 1., 0., 0.;
                               0., 0., 1., 2., 0., 0., 0., 1., 0.;
                               0., 0., 0., 0., 1., 2., 0., 0., 1.]);
    }
}
