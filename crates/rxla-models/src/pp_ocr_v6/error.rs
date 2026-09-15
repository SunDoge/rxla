use snafu::Snafu;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(super)))]
pub enum Error {
    #[snafu(display("PP-OCRv6 input images must have nonzero dimensions"))]
    EmptyImage,
    #[snafu(display("PP-OCRv6 target width/limit must be nonzero and valid"))]
    InvalidTarget,
    #[snafu(display("CTC shape does not match the probability buffer or blank token"))]
    InvalidCtcShape,
}
