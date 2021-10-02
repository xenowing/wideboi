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
        uniform_data: &[i32],
        num_threads: u32,
        expected_output_stream: &[i32],
    ) {
        println!("program: {:#?}", program);
        println!("expected output stream: {:?}", expected_output_stream);

        for (&binding, &stride) in input_stream_bindings.iter().zip(program.input_stream_thread_strides.iter()) {
            assert_eq!(binding.len() as u32, num_threads * stride);
        }

        assert_eq!(uniform_data.len() as u32, program.num_uniforms);

        println!("testing model");

        let (output_stream, num_cycles) = model(program, input_stream_bindings, uniform_data, num_threads);
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
            num_uniforms: 0,
            output_stream_thread_stride: 1,
        }, &[
            &input_stream,
        ],  &[],num_elements as _, &expected_output_stream);
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
            num_uniforms: 0,
            output_stream_thread_stride: 1,
        }, &[
            &input_stream,
        ],  &[],num_elements as _, &expected_output_stream);
    }

    #[test]
    fn uni_asm() {
        let instructions = vec![
            uni(R0, 0),
            sto(O0, R0, 0),
        ].into_iter().map(|i| i.encode()).collect();

        let num_elements = 10;

        let uniform_data = vec![1337];
        let expected_output_stream = vec![1337; num_elements as usize];

        test(&CompiledProgram {
            instructions,
            input_stream_thread_strides: Box::new([]),
            num_uniforms: 1,
            output_stream_thread_stride: 1,
        }, &[], &uniform_data, num_elements as _, &expected_output_stream);
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
        ], &[], num_elements as _, &expected_output_stream);

        Ok(())
    }

    #[test]
    fn x2_compiled() -> Result<(), CompileError> {
        let mut p = Program::new();

        let input = I0;
        let output = O0;
        let x = p.load_s(input);
        let res = p.add_s(x, x);
        p.store_s(output, res);

        let program = compile(&p)?;

        let num_elements = 10;

        let input_stream = (0..num_elements).into_iter().collect::<Vec<_>>();
        let expected_output_stream = (0..num_elements).into_iter().map(|x| x * 2).collect::<Vec<_>>();

        test(&program, &[
            &input_stream,
        ], &[], num_elements as _, &expected_output_stream);

        Ok(())
    }

    #[test]
    fn uni_compiled() -> Result<(), CompileError> {
        let mut p = Program::new();

        let uni = p.alloc_uni_s();
        let output = O0;
        let x = p.uni_s(uni);
        p.store_s(output, x);

        let program = compile(&p)?;

        let num_elements = 10;

        let uniform_data = vec![1337];
        let expected_output_stream = vec![1337; num_elements as usize];

        test(&program, &[], &uniform_data, num_elements as _, &expected_output_stream);

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
        ], &[], num_elements as _, &expected_output_stream);

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
        ], &[], num_elements as _, &expected_output_stream);

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
        ],  &[],num_elements as _, &expected_output_stream);

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
        ],  &[],num_elements as _, &expected_output_stream);

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
        let res = p.dot_v3(x, y, 0);
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
        ],  &[],num_elements as _, &expected_output_stream);

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
        let res = p.dot_v3(x, y, q_shift);
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
        ],  &[],num_elements as _, &expected_output_stream);

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
        ],  &[],num_elements as _, &expected_output_stream);

        Ok(())
    }

    #[test]
    fn single_m3_v3_mul_id() -> Result<(), CompileError> {
        let mut p = Program::new();

        let q_shift = 8;

        let input_m3 = I0;
        let input_v3 = I1;
        let output = O0;
        let m3 = p.load_m3(input_m3);
        let v3 = p.load_v3(input_v3);
        let res = p.mul_m3_v3(m3, v3, q_shift);
        p.store_v3(output, res);

        let program = compile(&p)?;

        let num_elements = 1;

        let input_stream_m4 = [
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        ].iter().map(|x| x << q_shift).collect::<Vec<_>>();
        let input_stream_v3 = vec![
            1, 2, 3,
        ].iter().map(|x| x << q_shift).collect::<Vec<_>>();
        let expected_output_stream = input_stream_v3.clone();

        test(&program, &[
            &input_stream_m4,
            &input_stream_v3,
        ], &[], num_elements, &expected_output_stream);

        Ok(())
    }

    #[test]
    fn single_m4_v4_mul_id() -> Result<(), CompileError> {
        let mut p = Program::new();

        let q_shift = 8;

        let input_m4 = I0;
        let input_v4 = I1;
        let output = O0;
        let m4 = p.load_m4(input_m4);
        let v4 = p.load_v4(input_v4);
        let res = p.mul_m4_v4(m4, v4, q_shift);
        p.store_v4(output, res);

        let program = compile(&p)?;

        let num_elements = 1;

        let input_stream_m4 = [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ].iter().map(|x| x << q_shift).collect::<Vec<_>>();
        let input_stream_v4 = vec![
            1, 2, 3, 4,
        ].iter().map(|x| x << q_shift).collect::<Vec<_>>();
        let expected_output_stream = input_stream_v4.clone();

        test(&program, &[
            &input_stream_m4,
            &input_stream_v4,
        ], &[], num_elements, &expected_output_stream);

        Ok(())
    }

    #[test]
    fn m4_uni_v4_muls_id() -> Result<(), CompileError> {
        let mut p = Program::new();

        let q_shift = 8;

        let uni_m4 = p.alloc_uni_m4();
        let input_v4 = I0;
        let output = O0;
        let m4 = p.uni_m4(uni_m4);
        let v4 = p.load_v4(input_v4);
        let res = p.mul_m4_v4(m4, v4, q_shift);
        p.store_v4(output, res);

        let program = compile(&p)?;

        let num_elements = 8;

        let input_stream_v4 = (0..num_elements * 4).map(|x| x << q_shift).collect::<Vec<_>>();
        let uniform_m4 = [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ].iter().map(|x| x << q_shift).collect::<Vec<_>>();
        let uniform_data = uniform_m4;
        let expected_output_stream = input_stream_v4.clone();

        test(&program, &[
            &input_stream_v4,
        ], &uniform_data, num_elements as _, &expected_output_stream);

        Ok(())
    }
}
