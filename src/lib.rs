#[macro_use]
extern crate static_assertions;

mod bit_array;
pub mod compiler;
pub mod instructions;
pub mod model;
pub mod program;

#[cfg(test)]
mod test {
    use super::compiler::*;
    use super::instructions::*;
    use super::instructions::InputStream::*;
    use super::instructions::OutputStream::*;
    use super::instructions::Register::*;
    use super::model::*;
    use super::program::*;

    pub fn test(
        program: &CompiledProgram,
        input_stream_bindings: &[&[i32]],
        num_threads: u32,
        expected_output_stream: &[i32],
    ) {
        println!("program: {:#?}", program);
        println!("expected output stream: {:?}", expected_output_stream);

        for (&binding, &stride) in input_stream_bindings.iter().zip(program.input_stream_thread_strides.iter()) {
            assert_eq!(binding.len() as u32, num_threads * stride);
        }

        println!("testing model");

        let (output_stream, num_cycles) = model(program, input_stream_bindings, num_threads);
        println!(" - output stream: {:?}", output_stream);
        println!(" - num cycles: {}", num_cycles);

        assert_eq!(output_stream, expected_output_stream);

        println!("success!");
    }

    #[test]
    fn id_asm() {
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
    fn x2_asm() {
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

    #[test]
    fn id_compiled() -> Result<(), CompileError> {
        let mut p = Program::new();

        let input = I0;
        let output = O0;
        let x = p.load_s(input);
        p.store_s(output, x);

        let program = compile(&p)?;

        let num_elements = 10;

        let input_stream = (0..num_elements).into_iter().collect::<Vec<_>>();
        let expected_output_stream = input_stream.clone();

        test(&program, &[
            &input_stream,
        ], num_elements as _, &expected_output_stream);

        Ok(())
    }

    #[test]
    fn x2_compiled() -> Result<(), CompileError> {
        let mut p = Program::new();

        let input = I0;
        let output = O0;
        let x = p.load_s(input);
        let res = p.add_s(x.clone(), x);
        p.store_s(output, res);

        let program = compile(&p)?;

        let num_elements = 10;

        let input_stream = (0..num_elements).into_iter().collect::<Vec<_>>();
        let expected_output_stream = (0..num_elements).into_iter().map(|x| x * 2).collect::<Vec<_>>();

        test(&program, &[
            &input_stream,
        ], num_elements as _, &expected_output_stream);

        Ok(())
    }

    #[test]
    fn scalar_sums() -> Result<(), CompileError> {
        let mut p = Program::new();

        let input_x = I0;
        let input_y = I1;
        let output = O0;
        let x = p.load_s(input_x);
        let y = p.load_s(input_y);
        let res = p.add_s(x, y);
        p.store_s(output, res);

        let program = compile(&p)?;

        let num_elements = 10;

        let input_stream_x = (0..num_elements).into_iter().collect::<Vec<_>>();
        let input_stream_y = (0..num_elements).into_iter().map(|x| x * 2).collect::<Vec<_>>();
        let expected_output_stream = (0..num_elements).into_iter().map(|x| x * 3).collect::<Vec<_>>();

        test(&program, &[
            &input_stream_x,
            &input_stream_y,
        ], num_elements as _, &expected_output_stream);

        Ok(())
    }

    #[test]
    fn scalar_muls_integer() -> Result<(), CompileError> {
        let mut p = Program::new();

        let input_x = I0;
        let input_y = I1;
        let output = O0;
        let x = p.load_s(input_x);
        let y = p.load_s(input_y);
        let res = p.mul_s(x, y, 0);
        p.store_s(output, res);

        let program = compile(&p)?;

        let num_elements = 10;

        let input_stream_x = (0..num_elements).into_iter().collect::<Vec<_>>();
        let input_stream_y = (0..num_elements).into_iter().map(|x| x * 2).collect::<Vec<_>>();
        let expected_output_stream = (0..num_elements).into_iter().map(|x| x * x * 2).collect::<Vec<_>>();

        test(&program, &[
            &input_stream_x,
            &input_stream_y,
        ], num_elements as _, &expected_output_stream);

        Ok(())
    }

    #[test]
    fn vector_sums() -> Result<(), CompileError> {
        let mut p = Program::new();

        let input_x = I0;
        let input_y = I1;
        let output = O0;
        let x = p.load_v3(input_x);
        let y = p.load_v3(input_y);
        let res = p.add_v3(x, y);
        p.store_v3(output, res);

        let program = compile(&p)?;

        let num_elements = 10;

        let input_stream_x = (0..num_elements).into_iter().flat_map(|x| [x * 1 + 0, x * 1 + 1, x * 1 + 2]).collect::<Vec<_>>();
        let input_stream_y = (0..num_elements).into_iter().flat_map(|x| [x * 2 + 0, x * 2 + 1, x * 2 + 2]).collect::<Vec<_>>();
        let expected_output_stream = (0..num_elements).into_iter().flat_map(|x| [x * 3 + 0, x * 3 + 2, x * 3 + 4]).collect::<Vec<_>>();

        test(&program, &[
            &input_stream_x,
            &input_stream_y,
        ], num_elements as _, &expected_output_stream);

        Ok(())
    }

    #[test]
    fn vector_muls_integer() -> Result<(), CompileError> {
        let mut p = Program::new();

        let input_x = I0;
        let input_y = I1;
        let output = O0;
        let x = p.load_v3(input_x);
        let y = p.load_v3(input_y);
        let res = p.mul_v3(x, y, 0);
        p.store_v3(output, res);

        let program = compile(&p)?;

        let num_elements = 10;

        let input_stream_x = (0..num_elements).into_iter().flat_map(|x| [x * 1 + 0, x * 1 + 1, x * 1 + 2]).collect::<Vec<_>>();
        let input_stream_y = (0..num_elements).into_iter().flat_map(|x| [x * 2 + 0, x * 2 + 1, x * 2 + 2]).collect::<Vec<_>>();
        let expected_output_stream =
            input_stream_x.iter()
            .zip(input_stream_y.iter())
            .map(|(&x, &y)| x * y)
            .collect::<Vec<_>>();

        test(&program, &[
            &input_stream_x,
            &input_stream_y,
        ], num_elements as _, &expected_output_stream);

        Ok(())
    }

    #[test]
    fn vector_dots_integer() -> Result<(), CompileError> {
        let mut p = Program::new();

        let input_x = I0;
        let input_y = I1;
        let output = O0;
        let x = p.load_v3(input_x);
        let y = p.load_v3(input_y);
        let res = p.dot(x, y, 0);
        p.store_s(output, res);

        let program = compile(&p)?;

        let num_elements = 10;

        let input_stream_x_v3 = (0..num_elements).into_iter().map(|x| [x * 1 + 0, x * 1 + 1, x * 1 + 2]).collect::<Vec<_>>();
        let input_stream_y_v3 = (0..num_elements).into_iter().map(|x| [x * 2 + 0, x * 2 + 1, x * 2 + 2]).collect::<Vec<_>>();
        let expected_output_stream =
            input_stream_x_v3.iter()
            .zip(input_stream_y_v3.iter())
            .map(|(x, y)| (x[0] * y[0]) + (x[1] * y[1]) + (x[2] * y[2]))
            .collect::<Vec<_>>();
        let input_stream_x = input_stream_x_v3.into_iter().flat_map(|x| x).collect::<Vec<_>>();
        let input_stream_y = input_stream_y_v3.into_iter().flat_map(|x| x).collect::<Vec<_>>();

        test(&program, &[
            &input_stream_x,
            &input_stream_y,
        ], num_elements as _, &expected_output_stream);

        Ok(())
    }

    #[test]
    fn vector_dots_q() -> Result<(), CompileError> {
        let mut p = Program::new();

        let q_shift = 16;

        let input_x = I0;
        let input_y = I1;
        let output = O0;
        let x = p.load_v3(input_x);
        let y = p.load_v3(input_y);
        let res = p.dot(x, y, q_shift);
        p.store_s(output, res);

        let program = compile(&p)?;

        let num_elements = 10;

        let input_stream_x_v3 = (0..num_elements).into_iter().map(|x| [
            (x * 1 + 0) << q_shift,
            (x * 1 + 1) << q_shift,
            (x * 1 + 2) << q_shift,
        ]).collect::<Vec<_>>();
        let input_stream_y_v3 = (0..num_elements).into_iter().map(|x| [
            (x * 2 + 0) << q_shift,
            (x * 2 + 1) << q_shift,
            (x * 2 + 2) << q_shift,
        ]).collect::<Vec<_>>();
        let expected_output_stream =
            input_stream_x_v3.iter()
            .zip(input_stream_y_v3.iter())
            .map(|(x, y)|
                (((x[0] as i64 * y[0] as i64) >> q_shift) as i32) +
                (((x[1] as i64 * y[1] as i64) >> q_shift) as i32) +
                (((x[2] as i64 * y[2] as i64) >> q_shift) as i32)
            )
            .collect::<Vec<_>>();
        let input_stream_x = input_stream_x_v3.into_iter().flat_map(|x| x).collect::<Vec<_>>();
        let input_stream_y = input_stream_y_v3.into_iter().flat_map(|x| x).collect::<Vec<_>>();

        test(&program, &[
            &input_stream_x,
            &input_stream_y,
        ], num_elements as _, &expected_output_stream);

        Ok(())
    }

    #[test]
    fn dead_scalar_loads() -> Result<(), CompileError> {
        let mut p = Program::new();

        let input = I0;
        let output = O0;
        let x = p.load_s(input);
        let y = p.load_s(input);
        let _z = p.load_s(input);
        let w = p.load_s(input);
        let lhs = p.add_s(x, y);
        let res = p.add_s(lhs, w);
        p.store_s(output, res);

        let program = compile(&p)?;

        let num_elements = 10;

        let input_stream = (0..num_elements * 4).into_iter().collect::<Vec<_>>();
        let expected_output_stream = (0..num_elements).map(|x| x * 4 * 3 + 1 + 3).into_iter().collect::<Vec<_>>();

        test(&program, &[
            &input_stream,
        ], num_elements as _, &expected_output_stream);

        Ok(())
    }
}
