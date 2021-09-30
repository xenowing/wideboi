#[macro_use]
extern crate static_assertions;

mod bit_array;
pub mod compiler;
pub mod instructions;
#[cfg(test)]
mod test_helpers;

#[cfg(test)]
mod test {
    use super::compiler::*;
    use super::instructions::*;
    use super::instructions::InputStream::*;
    use super::instructions::OutputStream::*;
    use super::instructions::Register::*;
    use super::test_helpers::*;

    #[test]
    fn id() {
        let instructions = vec![
            lod(R0, I0, 0),
            sto(O0, R0, 0),
        ].into_iter().map(|i| i.encode()).collect();

        let num_elements = 10;

        let input_stream = (0..num_elements).into_iter().collect::<Vec<_>>();
        let expected_output_stream = input_stream.clone();

        test(&CompiledProgram {
            instructions,
            input_stream_thread_strides: Box::new([1]),
            output_stream_thread_stride: 1,
        }, &[
            &input_stream,
        ], num_elements as _, &expected_output_stream);
    }

    #[test]
    fn x2() {
        let instructions = vec![
            lod(R0, I0, 0),
            add(R0, R0, R0),
            sto(O0, R0, 0),
        ].into_iter().map(|i| i.encode()).collect();

        let num_elements = 10;

        let input_stream = (0..num_elements).into_iter().collect::<Vec<_>>();
        let expected_output_stream = (0..num_elements).into_iter().map(|x| x * 2).collect::<Vec<_>>();

        test(&CompiledProgram {
            instructions,
            input_stream_thread_strides: Box::new([1]),
            output_stream_thread_stride: 1,
        }, &[
            &input_stream,
        ], num_elements as _, &expected_output_stream);
    }
}
