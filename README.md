# Floating Bar

This library provides the floating-bar type, which gives a memory-efficient representation of rational numbers. It is based on [Inigo Quilez's blog post exploring the concept](http://www.iquilezles.org/www/articles/floatingbar/floatingbar.htm).

For more information about the API and implementation details, please refer to the [library documentation](https://docs.rs/floating_bar/).

## Purpose

Almost all programming languages provide a way to represent different numeric types. These usually include natural numbers (`u16`), integers (`i32`), and reals (`f64`). However, there are no types that cover rational numbers. The purpose of this library is to fulfill this niche not covered by other numeric types.

Although floating-point types can usually serve as a good enough substitute, they can come at the cost of precision. Even small values run into this issue. For example, `1.0 - 0.8` evaluates to `0.19999999` for a 32-bit float. With floating bar, the equivalent expression is `1/1 - 8/10`, which evaluates to exactly `2/10`, or `0.2`.
<!--
## Performance

*Measurements for floats are done using hard floats.

| Tests (us) | f32  | r32  | f64  | r64  |
|------------|:----:|:----:|:----:|:----:|
| add        | 1.38 | 29.5 | 1.37 | 44.5 |
| mul        | 0.85 | 24.3 | 1.84 | 35.6 |
-->
## Features

### `std` (default)

This enables use of `std` when necessary. In particular, this implements `std::error::Error` for `ParseRatioErr`.

This feature is enabled by default.

### `denormals`

This enables denormal values. When the value of the denominator takes up the whole fraction field, the numerator will take an implicit value of 1.

Due to performance penalties, this is feature is disabled by default.

<!--
### `quiet-nan`

This feature enables using `NaN` as a return value for operations that would otherwise panic. These operations include: `Add`, `Sub`, `Mul`, `Div`, `.recip()`, and `.pow()`.

To encourage correct behavior and avoid latent bugs, this feature is disabled by default.
-->
## Contributing

Pull requests are welcome! Make sure that the documentation matches the functionality, and to add a unit test if possible.
<!--
### Todo

+ document how NaNs behave in comparison operations
  + is a total order desirable? useful?
+ document how conversion to float works w.r.t. precision
+ should remainder `%` be euclidian?
  + if not, should `%` follow Rust sign conventions?
-->
## License

This project is licensed under the [BSD Zero Clause License](https://choosealicense.com/licenses/0bsd/).
