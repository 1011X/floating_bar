# Floating Bar

This library provides the floating-bar type, which gives a memory-efficient representation for rational numbers. It is based on [Inigo Quilez's blog post exploring the concept](http://www.iquilezles.org/www/articles/floatingbar/floatingbar.htm).

For more information about the API and implementation details, please refer to the [library documentation](https://docs.rs/floating_bar/).

## Usage

To use this crate, add the following line to your `Cargo.toml`, under `[dependencies]`:
```toml
floating_bar = "0.4.0"
```

## Features

### `std` (default)

This enables use of `std` when necessary. In particular, this implements `std::error::Error` for `ParseRatioErr`.

This feature is enabled by default.

## Purpose

Almost all programming languages provide a way to represent different numeric types. These usually include natural numbers (`u16`), integers (`i32`), and reals (`f64`). However, there are no numeric types that cover rational numbers. The purpose of this library is to fulfill this niche not covered by other numeric types.

### Why use this over floating-point?

Although floating-point numbers can usually serve as a good enough substitute for rational numbers, this comes at the slight cost of precision due to the way they're encoded. This is immediately apparent when evaluating expressions that *should* be zero like `3/6 + 2/6 - 5/6`, which returns `-1.1102230246251565e-16` with double precision. Thus, floating-point numbers aren't always a good way to represent rational numbers.

Floating-bar numbers avoid this issue by storing the exact integer values in a compact format without losing precision. If your calculations involve handling fractions or dividing integers in general, this library will be a great fit.

### Why *not* use this over floating-point?

Conversely, this library is not meant as a replacement to floating-point numbers, just like rational numbers are not a replacement for real numbers. Floating-point is still beneficial when doing calculations where irrational numbers are involved (like *tau*, *pi*, *e*, or square roots), or when an approximate result is considered good enough.

That said, they're not mutually exclusive. There are conversion methods to convert from `r32` to `f32` when needed (and vice-versa).

## Contributing

Pull requests are welcome! Feel free to look at the issue list on GitHub for something you can work on. You can also submit an issue for a feature that would be useful to you.

## License

This project is licensed under the [BSD Zero Clause License](https://choosealicense.com/licenses/0bsd/).
