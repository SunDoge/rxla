use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{Expr, Ident, ItemFn, Local, Pat, Token, punctuated::Punctuated, visit_mut::VisitMut};

pub(super) fn expand(mut function: ItemFn, nn: TokenStream) -> syn::Result<TokenStream> {
    if function.sig.asyncness.is_some()
        || function.sig.constness.is_some()
        || function
            .sig
            .inputs
            .iter()
            .any(|arg| matches!(arg, syn::FnArg::Receiver(_)))
    {
        return Err(syn::Error::new_spanned(
            &function.sig,
            "#[model] requires a synchronous, non-const free function",
        ));
    }
    // Mixed-site identity is inaccessible to source bindings with the same spelling.
    let cx = Ident::new("__rxla_model_cx", Span::mixed_site());
    function
        .sig
        .inputs
        .insert(0, syn::parse_quote!(#cx: #nn::Cx));
    let mut visitor = Layers {
        cx,
        errors: None,
        nested: 0,
    };
    visitor.visit_block_mut(&mut function.block);
    if let Some(error) = visitor.errors {
        return Err(error);
    }
    Ok(quote!(#function))
}

struct Layers {
    cx: Ident,
    errors: Option<syn::Error>,
    nested: usize,
}

impl Layers {
    fn error(&mut self, error: syn::Error) {
        if let Some(errors) = &mut self.errors {
            errors.combine(error);
        } else {
            self.errors = Some(error);
        }
    }

    fn bind(&self, local: &mut Local, attribute: &syn::Attribute) -> syn::Result<()> {
        if self.nested != 0 {
            return Err(syn::Error::new_spanned(
                attribute,
                "#[layer] cannot capture context across a closure or nested item; use an explicit Cx there",
            ));
        }
        if !matches!(local.pat, Pat::Ident(_)) {
            return Err(syn::Error::new_spanned(
                &local.pat,
                "#[layer] requires an untyped single-variable let binding",
            ));
        }
        let segments =
            attribute.parse_args_with(Punctuated::<Expr, Token![,]>::parse_terminated)?;
        if segments.is_empty() {
            return Err(syn::Error::new_spanned(
                attribute,
                "#[layer] requires at least one explicit path segment",
            ));
        }
        for segment in &segments {
            if let Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(name),
                ..
            }) = segment
            {
                let value = name.value();
                if value.is_empty() || value.contains(['.', '/']) {
                    return Err(syn::Error::new_spanned(
                        name,
                        "layer names must be nonempty single path segments without '.' or '/'",
                    ));
                }
            }
        }
        let init = local.init.as_mut().ok_or_else(|| {
            syn::Error::new_spanned(attribute, "#[layer] requires a layer initializer")
        })?;
        if init.diverge.is_some() {
            return Err(syn::Error::new_spanned(
                attribute,
                "#[layer] does not support let-else",
            ));
        }
        let cx = &self.cx;
        let expression = &init.expr;
        let path = segments.iter().fold(
            quote!(#cx.clone()),
            |root, segment| quote!(#root.at(#segment)?),
        );
        // No closure, TLS, inference from variable names, or parameter creation.
        *init.expr = syn::parse_quote!((#path).layer(#expression));
        Ok(())
    }
}

impl VisitMut for Layers {
    fn visit_local_mut(&mut self, local: &mut Local) {
        let mut layer = None;
        local.attrs.retain(|attribute| {
            if !attribute.path().is_ident("layer") {
                return true;
            }
            if layer.is_some() {
                self.error(syn::Error::new_spanned(
                    attribute,
                    "duplicate #[layer] attribute",
                ));
            } else {
                layer = Some(attribute.clone());
            }
            false
        });
        // Walk the original initializer before inserting generated expressions.
        syn::visit_mut::visit_local_mut(self, local);
        if let Some(attribute) = layer
            && let Err(error) = self.bind(local, &attribute)
        {
            self.error(error);
        }
    }

    fn visit_expr_closure_mut(&mut self, expression: &mut syn::ExprClosure) {
        self.nested += 1;
        syn::visit_mut::visit_expr_closure_mut(self, expression);
        self.nested -= 1;
    }

    fn visit_item_mut(&mut self, item: &mut syn::Item) {
        self.nested += 1;
        syn::visit_mut::visit_item_mut(self, item);
        self.nested -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expand_source(source: &str) -> syn::Result<TokenStream> {
        expand(syn::parse_str(source)?, quote!(::rxla_nn))
    }

    #[test]
    fn expands_nested_loop_and_consumes_only_layer_attributes() -> syn::Result<()> {
        let tokens = expand_source(
            r#"fn model(x: Tensor) -> Result<Tensor> {
            for i in 0usize..2 {
                #[allow(unused_variables)]
                #[layer("blocks", i, "conv")]
                let conv = Conv2d::new(8, [3, 3]);
            }
            Ok(x)
        }"#,
        )?;
        let output = tokens.to_string();
        assert!(output.contains(":: rxla_nn :: Cx"));
        assert!(output.contains("at (\"blocks\") ? . at (i) ? . at (\"conv\") ?"));
        assert!(output.contains("allow (unused_variables)"));
        assert!(!output.contains("# [layer"));
        syn::parse2::<ItemFn>(tokens)?;
        Ok(())
    }

    #[test]
    fn rejects_invalid_layer_declarations() {
        for declaration in [
            "#[layer()] let x = Linear::new(2);",
            "#[layer(\"a.b\")] let x = Linear::new(2);",
            "#[layer(\"\")] let x = Linear::new(2);",
            "#[layer(\"a\")] #[layer(\"b\")] let x = Linear::new(2);",
            "#[layer(\"a\")] let (a, b) = pair;",
            "#[layer(\"a\")] let x: Linear = Linear::new(2);",
            "#[layer(\"a\")] let x;",
            "let f = || { #[layer(\"a\")] let x = Linear::new(2); };",
            "fn nested() { #[layer(\"a\")] let x = Linear::new(2); }",
        ] {
            assert!(
                expand_source(&format!("fn f() -> Result<()> {{ {declaration} Ok(()) }}")).is_err(),
                "{declaration}"
            );
        }
        assert!(expand_source("async fn f() {}").is_err());
        assert!(expand_source("const fn f() {}").is_err());
    }
}
