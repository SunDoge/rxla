//! Model syntax sugar; expanded code retains explicit context ownership.

use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::{Span, TokenStream as Tokens};
use quote::quote;
use syn::{ItemFn, parse_macro_input};

mod transform;

/// Inject a context parameter and bind `#[layer("name", index)]` declarations.
/// The expanded function is accepted directly by `Model::new`.
#[proc_macro_attribute]
pub fn model(args: TokenStream, input: TokenStream) -> TokenStream {
    if !args.is_empty() {
        return syn::Error::new(Span::call_site(), "#[model] takes no arguments")
            .into_compile_error()
            .into();
    }
    let function = parse_macro_input!(input as ItemFn);
    match nn_path().and_then(|path| transform::expand(function, path)) {
        Ok(output) => output.into(),
        Err(error) => error.into_compile_error().into(),
    }
}

fn nn_path() -> syn::Result<Tokens> {
    for (package, facade) in [("rxla-nn", false), ("rxla", true)] {
        if let Ok(found) = crate_name(package) {
            let root = match found {
                FoundCrate::Itself => quote!(crate),
                FoundCrate::Name(name) => {
                    let name = syn::Ident::new(&name, Span::call_site());
                    quote!(::#name)
                }
            };
            return Ok(if facade { quote!(#root::nn) } else { root });
        }
    }
    Err(syn::Error::new(
        Span::call_site(),
        "#[model] requires an rxla-nn or rxla dependency",
    ))
}
