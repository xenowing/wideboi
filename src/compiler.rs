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

    use std::ops::{Add, Mul};

    pub struct Program {
        pub statements: Vec<Statement>,
        pub variables: Vec<String>,
    }

    impl Program {
        pub fn new() -> Program {
            Program {
                statements: Vec::new(),
                variables: Vec::new(),
            }
        }

        pub fn load_s(&mut self, src: InputStream) -> Scalar {
            let name = self.temp_var();
            self.statements.push(Statement::Load(name.clone(), src));
            Scalar::VariableRef(name)
        }

        pub fn load_v3(&mut self, src: InputStream) -> V3 {
            V3 {
                x: self.load_s(src),
                y: self.load_s(src),
                z: self.load_s(src),
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

        fn temp_var(&mut self) -> String {
            let name = format!("__temp_{}", self.variables.len());
            self.variables.push(name.clone());
            name
        }
    }

    #[derive(Clone)]
    pub enum Scalar {
        Add(Box<Scalar>, Box<Scalar>),
        Multiply(Box<Scalar>, Box<Scalar>),
        VariableRef(String),
    }

    impl Add for Scalar {
        type Output = Self;

        fn add(self, other: Self) -> Self {
            Self::Add(Box::new(self), Box::new(other))
        }
    }

    impl Mul for Scalar {
        type Output = Self;

        fn mul(self, other: Self) -> Self {
            Self::Multiply(Box::new(self), Box::new(other))
        }
    }

    pub enum Statement {
        Load(String, InputStream),
        Store(OutputStream, Scalar),
    }

    #[derive(Clone)]
    pub struct V3 {
        pub x: Scalar,
        pub y: Scalar,
        pub z: Scalar,
    }

    impl V3 {
        pub fn dot(self, other: V3) -> Scalar {
            let temp = self * other;
            temp.x + temp.y + temp.z
        }
    }

    impl Add for V3 {
        type Output = Self;

        fn add(self, other: Self) -> Self {
            Self {
                x: self.x + other.x,
                y: self.y + other.y,
                z: self.z + other.z,
            }
        }
    }

    impl Mul for V3 {
        type Output = Self;

        fn mul(self, other: Self) -> Self {
            Self {
                x: self.x * other.x,
                y: self.y * other.y,
                z: self.z * other.z,
            }
        }
    }
}

// TODO: Move?
mod ir {
    use crate::instructions::*;

    #[derive(Debug)]
    pub struct Program {
        pub statements: Vec<Statement>,
        pub variables: Vec<Variable>,
    }

    impl Program {
        pub fn new() -> Program {
            Program {
                statements: Vec::new(),
                variables: Vec::new(),
            }
        }

        pub fn alloc_temp(&mut self) -> VariableIndex {
            let variable_index = self.variables.len();
            let name = format!("__temp_{}", variable_index);
            self.variables.push(Variable::new(name));
            VariableIndex(variable_index as _)
        }

        pub fn get_variable_index(&self, name: &str) -> Option<VariableIndex> {
            self.variables.iter().enumerate().find_map(|(i, v)| {
                if v.name == name { Some(VariableIndex(i as _)) } else { None }
            })
        }

        pub fn get_variable_register(&self, variable_index: VariableIndex) -> Option<Register> {
            self.variables[variable_index.0 as usize].register
        }

        pub fn get_variable_register_mut(&mut self, variable_index: VariableIndex) -> &mut Option<Register> {
            &mut self.variables[variable_index.0 as usize].register
        }
    }

    #[derive(Debug)]
    pub enum Statement {
        Add(VariableIndex, VariableIndex, VariableIndex),
        Load(VariableIndex, InputStream),
        Multiply(VariableIndex, VariableIndex, VariableIndex),
        Store(OutputStream, VariableIndex),
    }

    #[derive(Debug)]
    pub struct Variable {
        pub name: String,
        pub register: Option<Register>,
    }

    impl Variable {
        pub fn new(name: String) -> Variable {
            Variable {
                name,
                register: None,
            }
        }
    }

    #[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
    pub struct VariableIndex(pub u32);
}

pub fn compile(p: &program::Program) -> Result<Vec<EncodedInstruction>, CompileError> {
    let mut program = ir::Program::new();

    parse(p, &mut program);
    println!("Parse result: {:#?}", program);

    let (interference_edges, variable_degrees) = analyze_liveness(&program);
    println!("Interference edges: {:#?}", interference_edges);
    println!("Variable degrees: {:#?}", p.variables.iter().zip(&variable_degrees).collect::<Vec<_>>());

    allocate_registers(&mut program, interference_edges, variable_degrees)?;
    println!("Register allocation result: {:#?}", program.variables);

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
    for v in &p.variables {
        // TODO: Validity check(s) etc
        program.variables.push(ir::Variable::new(v.clone()));
    }

    for s in &p.statements {
        parse_statement(s, program);
    }
}

fn parse_statement(s: &program::Statement, program: &mut ir::Program) {
    match *s {
        program::Statement::Load(ref dst, src) => {
            program.statements.push(ir::Statement::Load(program.get_variable_index(dst).unwrap(), src))
        }
        program::Statement::Store(dst, ref src) => {
            let src = parse_scalar(src, program);
            program.statements.push(ir::Statement::Store(dst, src))
        }
    }
}

fn parse_scalar(s: &program::Scalar, program: &mut ir::Program) -> ir::VariableIndex {
    match *s {
        program::Scalar::Add(ref lhs, ref rhs) => {
            let lhs = parse_scalar(lhs, program);
            let rhs = parse_scalar(rhs, program);
            let t = program.alloc_temp();
            program.statements.push(ir::Statement::Add(t, lhs, rhs));
            t
        }
        program::Scalar::Multiply(ref lhs, ref rhs) => {
            let lhs = parse_scalar(lhs, program);
            let rhs = parse_scalar(rhs, program);
            let t = program.alloc_temp();
            program.statements.push(ir::Statement::Multiply(t, lhs, rhs));
            t
        }
        program::Scalar::VariableRef(ref src) => {
            program.get_variable_index(src).unwrap()
        }
    }
}

fn analyze_liveness(p: &ir::Program) -> (BTreeSet<(ir::VariableIndex, ir::VariableIndex)>, Vec<u32>) {
    let mut is_live = BitVector::new(p.variables.len());
    let mut interference_edges = BTreeSet::new();
    let mut variable_degrees = vec![0; p.variables.len()];

    for s in p.statements.iter().rev() {
        let (uses, def) = match *s {
            ir::Statement::Add(dst, lhs, rhs) => (vec![lhs, rhs], Some(dst)),
            ir::Statement::Multiply(dst, lhs, rhs) => (vec![lhs, rhs], Some(dst)),
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
        ir::Statement::Multiply(dst, lhs, rhs) => {
            let dst = program.get_variable_register(dst).unwrap();
            let lhs = program.get_variable_register(lhs).unwrap();
            let rhs = program.get_variable_register(rhs).unwrap();
            mul(dst, lhs, rhs)
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
        p.store_s(output, x.clone() + x);

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
        p.store_s(output, x + y);

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
        p.store_s(output, x * y);

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
        p.store_v3(output, x + y);

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
        p.store_v3(output, x * y);

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
        p.store_s(output, x.dot(y));

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
}
