use crate::bit_vector::*;
use crate::instructions::*;

use std::collections::BTreeSet;

use self::ir::VariableIndex;

#[derive(Debug)]
pub enum CompileError {
    // TODO: Proper desc/fields
    // TODO: Test(s)
    TooManyRegisters,
}

// TODO: Move?
mod program {
    use crate::instructions::*;

    pub struct Program {
        pub statements: Vec<Statement>,
        pub num_variables: u32,
    }

    impl Program {
        pub fn new() -> Program {
            Program {
                statements: Vec::new(),
                num_variables: 0,
            }
        }

        pub fn add_s(&mut self, lhs: Scalar, rhs: Scalar) -> Scalar {
            let t = self.alloc_var();
            self.statements.push(Statement::Add(t, lhs, rhs));
            Scalar::VariableRef(t)
        }

        pub fn add_v3(&mut self, lhs: V3, rhs: V3) -> V3 {
            V3 {
                x: self.add_s(lhs.x, rhs.x),
                y: self.add_s(lhs.y, rhs.y),
                z: self.add_s(lhs.z, rhs.z),
            }
        }

        fn alloc_var(&mut self) -> VariableIndex {
            let ret = VariableIndex(self.num_variables);
            self.num_variables += 1;
            ret
        }

        pub fn dot(&mut self, lhs: V3, rhs: V3, shift: u8) -> Scalar {
            let temp = self.mul_v3(lhs, rhs, shift);
            let lhs = self.add_s(temp.x, temp.y);
            self.add_s(lhs, temp.z)
        }

        pub fn load_s(&mut self, src: InputStream) -> Scalar {
            let name = self.alloc_var();
            self.statements.push(Statement::Load(name, src));
            Scalar::VariableRef(name)
        }

        pub fn load_v3(&mut self, src: InputStream) -> V3 {
            V3 {
                x: self.load_s(src),
                y: self.load_s(src),
                z: self.load_s(src),
            }
        }

        pub fn mul_s(&mut self, lhs: Scalar, rhs: Scalar, shift: u8) -> Scalar {
            let t = self.alloc_var();
            self.statements.push(Statement::Multiply(t, lhs, rhs, shift & SHIFT_FIELD_MASK));
            Scalar::VariableRef(t)
        }

        pub fn mul_v3(&mut self, lhs: V3, rhs: V3, shift: u8) -> V3 {
            V3 {
                x: self.mul_s(lhs.x, rhs.x, shift),
                y: self.mul_s(lhs.y, rhs.y, shift),
                z: self.mul_s(lhs.z, rhs.z, shift),
            }
        }

        pub fn store_s(&mut self, dst: OutputStream, src: Scalar) {
            self.statements.push(Statement::Store(dst, src));
        }

        pub fn store_v3(&mut self, dst: OutputStream, src: V3) {
            self.store_s(dst, src.x);
            self.store_s(dst, src.y);
            self.store_s(dst, src.z);
        }
    }

    #[derive(Clone)]
    pub enum Scalar {
        VariableRef(VariableIndex),
    }

    pub enum Statement {
        Add(VariableIndex, Scalar, Scalar),
        Load(VariableIndex, InputStream),
        Multiply(VariableIndex, Scalar, Scalar, u8),
        Store(OutputStream, Scalar),
    }

    #[derive(Clone)]
    pub struct V3 {
        pub x: Scalar,
        pub y: Scalar,
        pub z: Scalar,
    }

    #[derive(Clone, Copy)]
    pub struct VariableIndex(pub u32);
}

// TODO: Move?
mod ir {
    use super::*;

    #[derive(Debug)]
    pub struct Program {
        pub statements: Vec<Statement>,
        pub variable_registers: Vec<Option<Register>>,
    }

    impl Program {
        pub fn new() -> Program {
            Program {
                statements: Vec::new(),
                variable_registers: Vec::new(),
            }
        }

        pub fn get_variable_register(&self, variable_index: VariableIndex) -> Option<Register> {
            self.variable_registers[variable_index.0 as usize]
        }

        pub fn get_variable_register_mut(&mut self, variable_index: VariableIndex) -> &mut Option<Register> {
            &mut self.variable_registers[variable_index.0 as usize]
        }
    }

    #[derive(Debug)]
    pub enum Statement {
        Add(VariableIndex, VariableIndex, VariableIndex),
        Load(VariableIndex, InputStream),
        Multiply(VariableIndex, VariableIndex, VariableIndex, u8),
        Store(OutputStream, VariableIndex),
    }

    #[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
    pub struct VariableIndex(pub u32);

    impl From<program::VariableIndex> for VariableIndex {
        fn from(i: program::VariableIndex) -> Self {
            Self(i.0)
        }
    }
}

pub fn compile(p: &program::Program) -> Result<Vec<EncodedInstruction>, CompileError> {
    let mut program = ir::Program::new();

    parse(p, &mut program);
    println!("Parse result: {:#?}", program);

    let (interference_edges, variable_degrees) = analyze_liveness(&program);
    println!("Interference edges: {:#?}", interference_edges);
    println!("Variable degrees: {:#?}", variable_degrees.iter().enumerate().collect::<Vec<_>>());

    allocate_registers(&mut program, interference_edges, variable_degrees)?;
    println!("Variable registers: {:#?}", program.variable_registers);

    let instructions = generate_instructions(&program);
    println!("Instructions:");
    for instruction in &instructions {
        println!("  {}", instruction);
    }

    let encoded_instructions = encode_instructions(&instructions);
    println!("Encoded instructions:");
    for &encoded_instruction in &encoded_instructions {
        println!("  0x{:08x}", encoded_instruction);
    }

    Ok(encoded_instructions)
}

fn parse(p: &program::Program, program: &mut ir::Program) {
    program.variable_registers = vec![None; p.num_variables as usize];

    for s in &p.statements {
        parse_statement(s, program);
    }
}

fn parse_statement(s: &program::Statement, program: &mut ir::Program) {
    match *s {
        program::Statement::Add(dst, ref lhs, ref rhs) => {
            let lhs = parse_scalar(lhs);
            let rhs = parse_scalar(rhs);
            program.statements.push(ir::Statement::Add(dst.into(), lhs, rhs));
        }
        program::Statement::Load(dst, src) => {
            program.statements.push(ir::Statement::Load(dst.into(), src))
        }
        program::Statement::Multiply(dst, ref lhs, ref rhs, shift) => {
            let lhs = parse_scalar(lhs);
            let rhs = parse_scalar(rhs);
            program.statements.push(ir::Statement::Multiply(dst.into(), lhs, rhs, shift));
        }
        program::Statement::Store(dst, ref src) => {
            let src = parse_scalar(src);
            program.statements.push(ir::Statement::Store(dst, src))
        }
    }
}

fn parse_scalar(s: &program::Scalar) -> ir::VariableIndex {
    match *s {
        program::Scalar::VariableRef(src) => {
            src.into()
        }
    }
}

fn analyze_liveness(p: &ir::Program) -> (BTreeSet<(ir::VariableIndex, ir::VariableIndex)>, Vec<u32>) {
    let mut is_live = BitVector::new(p.variable_registers.len());
    let mut interference_edges = BTreeSet::new();
    let mut variable_degrees = vec![0; p.variable_registers.len()];

    for s in p.statements.iter().rev() {
        let (uses, def) = match *s {
            ir::Statement::Add(dst, lhs, rhs) => (vec![lhs, rhs], Some(dst)),
            ir::Statement::Multiply(dst, lhs, rhs, _) => (vec![lhs, rhs], Some(dst)),
            ir::Statement::Load(dst, _) => (Vec::new(), Some(dst)),
            ir::Statement::Store(_, src) => (vec![src], None),
        };
        if let Some(d) = def {
            is_live.clear(d.0 as _);
        }
        for u in uses {
            is_live.set(u.0 as _);
        }
        let mut is_live_x_iter = is_live.iter_set_indices();
        while let Some(x) = is_live_x_iter.next().map(|i| VariableIndex(i as _)) {
            let mut is_live_y_iter = is_live_x_iter.clone();
            while let Some(y) = is_live_y_iter.next().map(|i| VariableIndex(i as _)) {
                if interference_edges.insert((x, y)) {
                    variable_degrees[x.0 as usize] += 1;
                    variable_degrees[y.0 as usize] += 1;
                }
            }
        }
    }

    (interference_edges, variable_degrees)
}

fn allocate_registers(
    program: &mut ir::Program,
    interference_edges: BTreeSet<(ir::VariableIndex, ir::VariableIndex)>,
    variable_degrees: Vec<u32>,
) -> Result<(), CompileError> {
    let mut sorted_variables =
        variable_degrees.into_iter().enumerate().map(|(i, d)| (d, VariableIndex(i as _))).collect::<Vec<_>>();
    sorted_variables.sort_by_key(|&(d, _)| d);

    for v in sorted_variables.into_iter().map(|(_, v)| v).rev() {
        let mut register_index = 0;
        while interference_edges.iter().filter_map(|(x, y)| if *x == v {
            Some(*y)
        } else if *y == v {
            Some(*x)
        } else {
            None
        }).any(|other| {
            program
                .get_variable_register(other)
                .map(|r| r as u32 == register_index)
                .unwrap_or(false)
        }) {
            register_index += 1;
        }
        let register = Register::from_u32(register_index).ok_or(CompileError::TooManyRegisters)?;
        *program.get_variable_register_mut(v) = Some(register);
    }

    Ok(())
}

fn generate_instructions(program: &ir::Program) -> Vec<Instruction> {
    program.statements.iter().map(|s| match *s {
        ir::Statement::Add(dst, lhs, rhs) => {
            let dst = program.get_variable_register(dst).unwrap();
            let lhs = program.get_variable_register(lhs).unwrap();
            let rhs = program.get_variable_register(rhs).unwrap();
            add(dst, lhs, rhs)
        }
        ir::Statement::Multiply(dst, lhs, rhs, shift) => {
            let dst = program.get_variable_register(dst).unwrap();
            let lhs = program.get_variable_register(lhs).unwrap();
            let rhs = program.get_variable_register(rhs).unwrap();
            mul(dst, lhs, rhs, shift)
        }
        ir::Statement::Load(dst, src) => {
            let dst = program.get_variable_register(dst).unwrap();
            lod(dst, src)
        }
        ir::Statement::Store(dst, src) => {
            let src = program.get_variable_register(src).unwrap();
            sto(dst, src)
        }
    }).collect()
}

fn encode_instructions(instructions: &[Instruction]) -> Vec<EncodedInstruction> {
    instructions.iter().map(|&i| i.encode()).collect()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::compiler::program::Program;
    use crate::instructions::InputStream::*;
    use crate::instructions::OutputStream::*;
    use crate::test_helpers::*;

    #[test]
    fn id() -> Result<(), CompileError> {
        let mut p = Program::new();

        let input = I0;
        let output = O0;
        let x = p.load_s(input);
        p.store_s(output, x);

        let instructions = compile(&p)?;

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

        Ok(())
    }

    #[test]
    fn x2() -> Result<(), CompileError> {
        let mut p = Program::new();

        let input = I0;
        let output = O0;
        let x = p.load_s(input);
        let res = p.add_s(x.clone(), x);
        p.store_s(output, res);

        let instructions = compile(&p)?;

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

        let instructions = compile(&p)?;

        let num_elements = 10;

        let input_stream_x = (0..num_elements).into_iter().collect::<Vec<_>>();
        let input_stream_y = (0..num_elements).into_iter().map(|x| x * 2).collect::<Vec<_>>();
        let expected_output_stream = (0..num_elements).into_iter().map(|x| x * 3).collect::<Vec<_>>();

        test(&instructions, &[
            InputStreamInfo {
                data: &input_stream_x,
                thread_stride: 1,
            },
            InputStreamInfo {
                data: &input_stream_y,
                thread_stride: 1,
            },
        ], OutputStreamInfo {
            num_words: num_elements as _,
            thread_stride: 1,
        }, num_elements as _, &expected_output_stream);

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

        let instructions = compile(&p)?;

        let num_elements = 10;

        let input_stream_x = (0..num_elements).into_iter().collect::<Vec<_>>();
        let input_stream_y = (0..num_elements).into_iter().map(|x| x * 2).collect::<Vec<_>>();
        let expected_output_stream = (0..num_elements).into_iter().map(|x| x * x * 2).collect::<Vec<_>>();

        test(&instructions, &[
            InputStreamInfo {
                data: &input_stream_x,
                thread_stride: 1,
            },
            InputStreamInfo {
                data: &input_stream_y,
                thread_stride: 1,
            },
        ], OutputStreamInfo {
            num_words: num_elements as _,
            thread_stride: 1,
        }, num_elements as _, &expected_output_stream);

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

        let instructions = compile(&p)?;

        let num_elements = 10;

        let input_stream_x = (0..num_elements).into_iter().flat_map(|x| [x * 1 + 0, x * 1 + 1, x * 1 + 2]).collect::<Vec<_>>();
        let input_stream_y = (0..num_elements).into_iter().flat_map(|x| [x * 2 + 0, x * 2 + 1, x * 2 + 2]).collect::<Vec<_>>();
        let expected_output_stream = (0..num_elements).into_iter().flat_map(|x| [x * 3 + 0, x * 3 + 2, x * 3 + 4]).collect::<Vec<_>>();

        test(&instructions, &[
            InputStreamInfo {
                data: &input_stream_x,
                thread_stride: 3,
            },
            InputStreamInfo {
                data: &input_stream_y,
                thread_stride: 3,
            },
        ], OutputStreamInfo {
            num_words: (num_elements * 3) as _,
            thread_stride: 3,
        }, num_elements as _, &expected_output_stream);

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

        let instructions = compile(&p)?;

        let num_elements = 10;

        let input_stream_x = (0..num_elements).into_iter().flat_map(|x| [x * 1 + 0, x * 1 + 1, x * 1 + 2]).collect::<Vec<_>>();
        let input_stream_y = (0..num_elements).into_iter().flat_map(|x| [x * 2 + 0, x * 2 + 1, x * 2 + 2]).collect::<Vec<_>>();
        let expected_output_stream =
            input_stream_x.iter()
            .zip(input_stream_y.iter())
            .map(|(&x, &y)| x * y)
            .collect::<Vec<_>>();

        test(&instructions, &[
            InputStreamInfo {
                data: &input_stream_x,
                thread_stride: 3,
            },
            InputStreamInfo {
                data: &input_stream_y,
                thread_stride: 3,
            },
        ], OutputStreamInfo {
            num_words: (num_elements * 3) as _,
            thread_stride: 3,
        }, num_elements as _, &expected_output_stream);

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

        let instructions = compile(&p)?;

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

        test(&instructions, &[
            InputStreamInfo {
                data: &input_stream_x,
                thread_stride: 3,
            },
            InputStreamInfo {
                data: &input_stream_y,
                thread_stride: 3,
            },
        ], OutputStreamInfo {
            num_words: num_elements as _,
            thread_stride: 1,
        }, num_elements as _, &expected_output_stream);

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

        let instructions = compile(&p)?;

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

        test(&instructions, &[
            InputStreamInfo {
                data: &input_stream_x,
                thread_stride: 3,
            },
            InputStreamInfo {
                data: &input_stream_y,
                thread_stride: 3,
            },
        ], OutputStreamInfo {
            num_words: num_elements as _,
            thread_stride: 1,
        }, num_elements as _, &expected_output_stream);

        Ok(())
    }
}
