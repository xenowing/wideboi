mod bit_vector;
pub mod compiler;
mod instructions;
#[cfg(test)]
mod test_helpers;

#[cfg(test)]
mod test {
    use super::instructions::*;
    use super::instructions::InputStream::*;
    use super::instructions::OutputStream::*;
    use super::instructions::Register::*;
    use super::test_helpers::*;

    #[test]
    fn id() {
        let instructions = vec![
            lod(R0, I0),
            str(O0, R0),
        ];

        let num_elements = 10;

        let input_stream = (0..num_elements).into_iter().collect::<Vec<_>>();
        let expected_output_stream = input_stream.clone();

        test(&instructions, &[
            InputStreamInfo {
                data: &input_stream,
                thread_stride: 1,
            },
        ], OutputStreamInfo {
            num_words: num_elements as _,
            thread_stride: 1,
        }, num_elements as _, &expected_output_stream);
    }

    #[test]
    fn x2() {
        let instructions = vec![
            lod(R0, I0),
            add(R0, R0, R0),
            str(O0, R0),
        ];

        let num_elements = 10;

        let input_stream = (0..num_elements).into_iter().collect::<Vec<_>>();
        let expected_output_stream = (0..num_elements).into_iter().map(|x| x * 2).collect::<Vec<_>>();

        test(&instructions, &[
            InputStreamInfo {
                data: &input_stream,
                thread_stride: 1,
            },
        ], OutputStreamInfo {
            num_words: num_elements as _,
            thread_stride: 1,
        }, num_elements as _, &expected_output_stream);
    }
}
