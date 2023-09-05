use proc_macro::TokenStream;
use quote::quote;

fn get_layers_and_ident(input: syn::DeriveInput) ->
    (syn::Type, syn::Ident)
{
    let ident = input.ident;
    if let syn::Data::Struct(data) = input.data {
        if let syn::Fields::Named(fields) = data.fields {
            if fields.named.len() != 1 {
                panic!("Could not construct neural \
                        network: {} struct has too \
                        many fields!", ident);
            }

            let layers_field = fields.named
                .into_iter().next().unwrap();
            let field_ident = layers_field.ident;
            if field_ident.is_none() {
                panic!("Could not construct neural network: \
                        {} struct does not have `layers` \
                        field!", ident);
            }
            let fieldname = field_ident
                .unwrap().to_string();

            if String::from("layers") != fieldname {
                panic!("Could not construct neural network: \
                        {} struct does not have `layers` \
                        field!", ident);
            }

            return (layers_field.ty, ident);
        } else {
            panic!("Could not construct neural network: \
                    {} struct does not have `layers` \
                    field!", ident);
        }
    } else {
        panic!("Could not construct neural network: \
                {} is not a struct!", ident);
    }
}

#[proc_macro_attribute]
pub fn neural_network(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input: syn::DeriveInput =
        syn::parse(item.clone()).unwrap();
    let (ty, ident) = get_layers_and_ident(input);

    let layers = if let syn::Type::Tuple(layers) = ty {
        layers.elems
    } else {
        panic!("Could not construct neural network: \
                `layers` is not a tuple!");
    };

    let mut layer_idents: Vec<syn::Path> = Vec::new();
    for layer in layers.into_iter() {
        layer_idents.push(match layer {
            syn::Type::Path(path) => path.path,
            _ => panic!("Could not construct neural network: \
                    Invalid layer type!"),
        });
    }
    
    let layer_idents: Vec<syn::Path> = layer_idents
        .into_iter().collect();

    let first_layer = &layer_idents[0];
    let last_layer = &layer_idents[
        layer_idents.len()-1];

    let mut new_list = proc_macro2::TokenStream::new();
    for layer_ident in layer_idents.iter() {
        new_list.extend(quote!{
            #layer_ident::new(),
        });
    }

    let mut layers_string = String::from("[");
    for i in 0..layer_idents.len() {
        let layer_ident = &layer_idents[i];
        layers_string += "\"";
        layers_string += &quote!(#layer_ident).to_string();
        layers_string += "\"";
        if i < layer_idents.len()-1 {
            layers_string += ", ";
        }
    }
    layers_string += "]";

    let mut params_cnt_sum = proc_macro2::TokenStream::new();
    for i in 0..layer_idents.len() {
        let layer_ident = &layer_idents[i];
        params_cnt_sum.extend(quote!{#layer_ident::PARAMS_CNT});
        if i < layer_idents.len()-1 {
            params_cnt_sum.extend(quote!{ + });
        }
    }

    let mut d_offsets: Vec<proc_macro2::TokenStream> =
        Vec::with_capacity(layer_idents.len()+1);
    d_offsets.push(quote!{0});
    for i in 0..layer_idents.len() {
        let layer_ident = &layer_idents[i];
        d_offsets.push(d_offsets[i].clone());
        d_offsets[i+1].extend(quote!{
            + #layer_ident::PARAMS_CNT
        });
    }

    let mut eval_all_layers = proc_macro2::TokenStream::new();
    for i in 0..layer_idents.len() {
        let layer_ident = &layer_idents[i];

        let old_offset = &d_offsets[i];
        let offset = &d_offsets[i+1];
        eval_all_layers.extend(quote!{
            let x = unsafe {
                #layer_ident::eval_unchecked(
                &p[#old_offset..#offset], x)
            };
        });
    }

    let mut forward_all_layers = proc_macro2::TokenStream::new();
    let mut backward_all_layers = proc_macro2::TokenStream::new();
    for i in 0..layer_idents.len() {
        let old_offset = &d_offsets[i];
        let offset = &d_offsets[i+1];
        let idx: syn::Index = i.into();

        forward_all_layers.extend(quote!{
            let x = self.layers.#idx.forward(
                &p[#old_offset..#offset], x);
        });
        backward_all_layers.extend(quote!{
            self.layers.#idx.backward(
                &p[#old_offset..#offset]);
        });
    }

    let mut compute_jacobian = proc_macro2::TokenStream::new();
    compute_jacobian.extend(quote!{
        let mut jm: DMatrix<f64> =
            DMatrix::from_element_generic(
                nalgebra::base::dimension::Dyn(Self::NEURONS_OUT),
                nalgebra::base::dimension::Dyn(Self::PARAMS_CNT), 0f64);
        let m: DMatrix<f64> = {
            let mut m: DMatrix<f64> =
                DMatrix::from_element_generic(
                    nalgebra::base::dimension::Dyn(Self::NEURONS_OUT),
                    nalgebra::base::dimension::Dyn(Self::NEURONS_OUT), 0f64);
            m.fill_diagonal(1f64);

            m
        };
        let mut offset: usize = Self::PARAMS_CNT;
    });
    for i in (1..layer_idents.len()).rev() {
        let layer_ident = &layer_idents[i];
        let idx: syn::Index = i.into();
        let prev_idx: syn::Index = (i-1).into();
        compute_jacobian.extend(quote!{
            let jf = &m * self.layers.#idx.chain_end(
                    &self.layers.#prev_idx.signal);
            offset -= #layer_ident::PARAMS_CNT;
            for i in offset..offset+#layer_ident::PARAMS_CNT {
                jm.set_column(i, &jf.index((.., i - offset)));
            }
            
            let m = m * self.layers.#idx.chain_element();
        });
    }
    let idx: syn::Index = 0.into();
    compute_jacobian.extend(quote!{
        let jf = m * self.layers.#idx.chain_end(&x);
        for i in 0..#first_layer::PARAMS_CNT {
            jm.set_column(i, &jf.index((.., i)));
        }
    });

    let network_trait_impl = quote! {
        impl Network for #ident {
            const PARAMS_CNT: usize = #params_cnt_sum;
            const NEURONS_IN: usize = #first_layer::NEURONS_IN;
            const NEURONS_OUT: usize = #last_layer::NEURONS_OUT;

            fn new() -> Self {
                Self {
                    layers: (#new_list),
                }
            }

            fn layers_info() -> &'static str {
                #layers_string
            }

            fn eval(p: &[f64], x: DVector<f64>) ->
                DVector<f64>
            {
                assert_eq!(p.len(), Self::PARAMS_CNT);
                assert_eq!(x.len(), Self::NEURONS_IN);

                #eval_all_layers

                x
            }

            fn forward(&mut self, p: &[f64], x: DVector<f64>) ->
                DVector<f64>
            {
                assert_eq!(p.len(), Self::PARAMS_CNT);
                assert_eq!(x.len(), Self::NEURONS_IN);

                #forward_all_layers

                x
            }

            fn backward(&mut self, p: &[f64])
            {
                assert_eq!(p.len(), Self::PARAMS_CNT);

                #backward_all_layers
            }

            fn jacobian(&mut self, x: &DVector<f64>) ->
                DMatrix<f64>
            {
                assert_eq!(x.len(), Self::NEURONS_IN);

                #compute_jacobian

                jm
            }
        }
    };

    let mut output =
        proc_macro2::TokenStream::from(item);
    output.extend(network_trait_impl);

    proc_macro::TokenStream::from(output)
}
