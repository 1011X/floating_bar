# Floating Bar

This library provides the floating bar type, which allows for efficient
representation of rational numbers without loss of precision. It is based on
[Inigo Quilez's blog post exploring the concept](http://www.iquilezles.org/www/articles/floatingbar/floatingbar.htm).

For more information, refer to the [library documentation](https://docs.rs/floating_bar/).

## Features

### `denormals`

This enables denormal values. When the value of the denominator takes up the whole fraction field, the numerator will take an implicit value of 1.

Due to the performance penalty of calculating with denormal values, this is disabled by default.

## Contributing

Pull requests welcome! There are plenty of `todo!()` methods that one can fill in. Make sure that the documentation matches the functionality, and to add a unit test for it. If you don't wish to write a test, just add the function for it and leave it for someone else to fill it in.

## License

This project is licensed under the [BSD Zero Clause License](https://choosealicense.com/licenses/0bsd/).
